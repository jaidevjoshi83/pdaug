import matplotlib
matplotlib.use('Agg')
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import quantiprot
from quantiprot.utils.io import load_fasta_file
from quantiprot.utils.feature import Feature, FeatureSet
from quantiprot.metrics.aaindex import get_aa2volume, get_aa2hydropathy
from quantiprot.metrics.basic import average
# Local Fisher-test related imports:
#from quantiprot.analysis.fisher import local_fisher_2d, _plot_local_fisher_2d
from matplotlib import pyplot as plt


from math import log10, floor
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import fisher_exact
from quantiprot.utils.sequence import SequenceSet, compact


def _count_frame(data, frame_range, num_bins):
    """
    Count instances in a 2D frame

    The function discretizes the feature space into a grid of cells.
    Then it counts the number of instances that fall into each cell.
    An efficient method for counting instances is used. It performs parallel
    logical comparisons of data instances to vectors that hold information on
    grid lines.

    Args:
        data (numpy.matrix): a Nx2 data matrix
        frame_range (numpy.matrix): a 2x2 matrix which defines feature ranges
        num_bins (list): a pair defining the resolution of the 2D grid
    Returns:
        cell_counts (numpy.matrix): a matrix holding counts of instances in
            each grid cell
        bin_ranges (tuple): a pair of numpy matrices holding information on
            bin(grid_cell) ranges
    """
    grid_x = np.linspace(start=frame_range[0, 0], stop=frame_range[1, 0],\
                          num=num_bins[0]+1, endpoint=True)
    grid_y = np.linspace(start=frame_range[0, 1], stop=frame_range[1, 1],\
                          num=num_bins[1]+1, endpoint=True)
    # copy because we add ones in the next lines
    bin_ranges = (np.copy(grid_x), np.copy(grid_y))


    #Count points in each grid cell
    grid_x[-1] += 1 # the last cell has to contain data at the border
    grid_y[-1] += 1 # the last cell has to contain data at the border

    gte_x = np.matrix(data[:, 0] >= grid_x, dtype='float64')
    lt_x = np.matrix(data[:, 0] < grid_x, dtype='float64')
    gte_y = np.matrix(data[:, 1] >= grid_y, dtype='float64')
    lt_y = np.matrix(data[:, 1] < grid_y, dtype='float64')

    dif_x = gte_x - lt_x
    dif_y = gte_y - lt_y

    bins_x = dif_x.argmin(axis=1) - 1
    bins_y = dif_y.argmin(axis=1) - 1

    coords = np.concatenate((bins_x, bins_y), axis=1)

    cell_counts = np.zeros(shape=(len(grid_x)-1, len(grid_y)-1))

    for i in range(coords.shape[0]):
        cell_counts[coords[i, 0], coords[i, 1]] += 1

    return cell_counts, bin_ranges


def local_fisher_2d(set1, set2, features=None, \
                    windows_per_frame=10, overlap_factor=1, frame_range=None):
    """
    Compare local and global distribution of samples from two populations
    in the 2d feature space using the Fisher's exact test.

    The function performs the Fisher Exact Test for comparing local and global
    ratia of instance counts from two different populations. It uses the
    '_count_frame' function to discretize the feature space and get instance
    counts. Then it scans the 2d feature space with a sliding window and
    performs the Fisher Exact test.

        Args:
            set1 (SequenceSet or numpy.matrix): the first set with at least
                2 sequence features.
            set2 (SequenceSet or numpy.matrix): the second set with at least
                2 sequence features.
            features (tuple or list): strings with feature names for running
                the 2d Fisher test. If None then the first two features are
                used. Relevant only if 'set1' or 'set2' are SequenceSets.
            windows_per_frame (int): ratio between the whole feature space and
                the sliding window (default 10).
            overlap_factor (int):ratio between the size of a sliding window
                and a discretization grid cell (default 1).
            frame_range(numpy.matrix): 2x2 matrix with range of features
                in both dimensions.

        Returns final_res (dict): a dictionary including:
            'odds_ratio' (numpy.matrix): a matrix of odds_ratios obtained
                in each sliding window position.
            'p_value' (numpy.matrix): a matrix containing Fisher test outcome
                pvalues in each sliding window position.
            'w_counts1' (numpy.matrix): a matrix with first population instance
                counts in each sliding window position.
            'w_counts2' (numpy.matrix): a matrix with second population instance
                counts in each sliding window position.
            'w_center_x' (numpy.matrix): matrix containing coordinates of window
                centers in the X dimension.
            'w_center_y' (numpy.matrix): matrix containing coordinates of window
                centers in the Y dimension.
            '_bin_ranges_x' (numpy.matrix): matrix containing bin(grid_cell)
                ranges in the X dimension.
            '_bin_ranges_y' (numpy.matrix): matrix containing bin(grid_cell)
                ranges in the Y dimension.
    """

    if isinstance(set1, SequenceSet):
        mat1 = np.transpose(np.matrix(compact(set1,
                                              features=features).columns()))
    if isinstance(set2, SequenceSet):
        mat2 = np.transpose(np.matrix(compact(set2,
                                              features=features).columns()))

    #Deal with window_per_frame and overlap_factor
    #given either as a scalar or as a list-like
    if not hasattr(windows_per_frame, "__len__"):
        w_per_frame = (windows_per_frame, windows_per_frame)
    else:
        w_per_frame = (windows_per_frame[0], windows_per_frame[1])

    if not hasattr(overlap_factor, "__len__"):
        w_size = (overlap_factor, overlap_factor)
    else:
        w_size = (overlap_factor[0], overlap_factor[1])

    num_bins = (w_per_frame[0]*w_size[0], w_per_frame[1]*w_size[1])

    if frame_range is None:
        #Evaluate the range of features in both populations.

        frame_range = np.concatenate((np.minimum(mat1.min(0), mat2.min(0)),\
                                      np.maximum(mat1.max(0), mat2.max(0))))

        margin_x = (frame_range[1, 0] - frame_range[0, 0])/w_per_frame[0]
        margin_y = (frame_range[1, 1] - frame_range[0, 1])/w_per_frame[1]

        frame_range[0, 0] -= margin_x
        frame_range[1, 0] += margin_x

        frame_range[0, 1] -= margin_y
        frame_range[1, 1] += margin_y

    #Discretize feature space into NxM grid,
    #where N = w_per_frame[0]*w_size[0].
    #      M = w_per_frame[1]*w_size[1].
    #count instances of population1 and population2 in each grid cell.
    #both bin ranges are always the same because the frame range is common.
    cell_counts1, bin_ranges = _count_frame(mat1, frame_range=frame_range,\
                                          num_bins=num_bins)
    cell_counts2, _ = _count_frame(mat2, frame_range=frame_range,\
                                          num_bins=num_bins)

    #Number of windows that fit in a single row/column of a frame
    w_number = (cell_counts1.shape[0]-w_size[0]+1,
                cell_counts1.shape[1]-w_size[1]+1)

    #Initialize matrices holding counts at scanning window positions.
    window_counts1 = np.zeros(shape=w_number)
    window_counts2 = np.zeros(shape=w_number)

    #Initialize matrices holding window coordinates
    window_center_x = np.zeros(shape=w_number[0])
    window_center_y = np.zeros(shape=w_number[1])

    #Initialize matrices holding Fisher Exact test results
    fisher_pv = np.ones(shape=w_number)
    odds_ratio = np.ones(shape=w_number)

    #Calculate population totals in the whole feature space
    all1 = cell_counts1.sum()
    all2 = cell_counts2.sum()

    #Calculate window centers
    for start_x in range(0, w_number[0]):
        window_center_x[start_x] = (bin_ranges[0][start_x]+ \
                                    bin_ranges[0][start_x+w_size[0]])/2
    for start_y in range(0, w_number[1]):
        window_center_y[start_y] = (bin_ranges[1][start_y]+ \
                                    bin_ranges[1][start_y+w_size[1]])/2

    #Scan the feature space with a step of 1 cell.
    for start_x in range(0, w_number[0]):

        for start_y in range(0, w_number[1]):
            #Count instances of each population in the window
            window_counts1[start_x, start_y] = \
                cell_counts1[start_x:(start_x+w_size[0]), \
                             start_y:(start_y+w_size[1])].sum()
            window_counts2[start_x, start_y] = \
                cell_counts2[start_x:(start_x+w_size[0]), \
                             start_y:(start_y+w_size[1])].sum()
            #Perform the Fisher Exact Test against
            #h0: population ratio in the window the same as in the whole space.
            odds_ratio[start_x, start_y], fisher_pv[start_x, start_y] =\
                fisher_exact([[all1, window_counts1[start_x, start_y]],\
                              [all2, window_counts2[start_x, start_y]]])

    fisher_res = {'p_value':fisher_pv, 'odds_ratio':odds_ratio,\
                'w_counts1':window_counts1, 'w_counts2':window_counts2,\
                'w_center_x':window_center_x, 'w_center_y':window_center_y,\
                '_bin_ranges_x':bin_ranges[0], '_bin_ranges_y':bin_ranges[1]}

    return fisher_res


def _plot_local_fisher_2d(fisher_res, xlabel="feat_1", ylabel="feat_2",
                          pop1_label="pop_1", pop2_label="pop_2", out_file_path=None, fig_width=8, fig_hight=8, fig_hspace=0.35, fig_wspace=0.25):
    """
    Plot results of the local Fisher's extact test in the 2d space.

    Args:
        fisher_res (dict): output from 'fisher_local_2d'.
        xlabel (str): name of the 1st feature to appear in the plots
            (default: "feat_1")
        ylabel (str): name of the 2nd feature to appear in the plots
            (default: "feat_2")
        pop1_label (str): name of the 1st population to appear in the plots
            (default: "pop_1")
        pop2_label (str): name of the 2nd population to appear in the plots
            (default: "pop_2")
    """
    fisher_or = fisher_res["odds_ratio"]
    fisher_c1 = fisher_res["w_counts1"]
    fisher_c2 = fisher_res["w_counts2"]
    fisher_pv = fisher_res["p_value"]

    for pos_x in range(len(fisher_or)):
        for pos_y in range(len(fisher_or[0])):
            if fisher_c1[pos_x][pos_y] == 0 and fisher_c2[pos_x][pos_y] == 0:
                fisher_or[pos_x][pos_y] = np.nan
            elif fisher_c1[pos_x][pos_y] == 0:
                fisher_or[pos_x][pos_y] = np.inf
            elif fisher_c2[pos_x][pos_y] == 0:
                fisher_or[pos_x][pos_y] = -np.inf
            elif fisher_or[pos_x][pos_y] < 1:
                fisher_or[pos_x][pos_y] = -1.0/fisher_or[pos_x][pos_y]

    vmax_abs = np.nanmax(np.abs([x for x in np.array(fisher_or).flatten()
                                 if x > -np.inf and x < np.inf]))

    for pos_x in range(len(fisher_or)):
        for pos_y in range(len(fisher_or[0])):
            if abs(fisher_or[pos_x][pos_y]) == np.inf:
                fisher_or[pos_x][pos_y] = np.sign(fisher_or[pos_x][pos_y])*vmax_abs

    ##### Extra Fig perimeters added ################################
    plt.figure(figsize=(fig_width, fig_hight)) # Figure size 
    plt.subplots_adjust(hspace = fig_hspace, wspace = fig_wspace) # space between the subplots. 
    ##################################################################

    plt.subplot(221)
    plt.pcolormesh(fisher_res["w_center_x"], fisher_res["w_center_y"],
                   np.ma.masked_invalid(fisher_c1).T, cmap="Reds")
    plt.colorbar()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Counts "+pop1_label)

    plt.subplot(222)
    plt.pcolormesh(fisher_res["w_center_x"], fisher_res["w_center_y"],
                   np.ma.masked_invalid(fisher_c2).T, cmap="Reds")
    plt.colorbar()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Counts "+pop2_label)

    cmap = plt.get_cmap('RdBu')
    cmap.set_bad(color='k', alpha=1.)

    cbar_lo = 1.0/vmax_abs
    cbar_lo_places = max(0, -floor(log10(cbar_lo))+1)
    cbar_hi = vmax_abs
    cbar_hi_places = max(0, -floor(log10(cbar_hi))+1)

    plt.subplot(223)
    plt.pcolormesh(fisher_res["w_center_x"], fisher_res["w_center_y"],
                   np.ma.masked_invalid(fisher_or).T, cmap=cmap,
                   vmin=-vmax_abs, vmax=vmax_abs)
    cbar = plt.colorbar(ticks=([-vmax_abs, 0, vmax_abs]))
    cbar.ax.set_yticklabels(['< '+str(round(cbar_lo, int(cbar_lo_places))), '1',
                             '> '+str(round(cbar_hi, int(cbar_hi_places)))])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Odds ratio")

    plt.subplot(224)
    plt.pcolormesh(fisher_res["w_center_x"], fisher_res["w_center_y"],
                   np.log10(np.ma.masked_invalid(fisher_pv)).T, cmap="RdGy")
    plt.colorbar()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Fisher test\np-value (logarithm of 10)")

    #Savefig function added with preserving default behavior

    if out_file_path==None:
        plt.show()
    else:
        plt.savefig(out_file_path,dpi=300)


def HTML_Gen(html):

    out_html = open(html,'w')             
    part_1 =  """

    <!DOCTYPE html>
    <html lang="en">
    <head>
      <title>Bootstrap Example</title>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.0/css/bootstrap.min.css">
      <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.0/jquery.min.js"></script>
      <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.0/js/bootstrap.min.js"></script>
    <body>
    <style>
    div.container_1 {
      width:600px;
      margin: auto;
     padding-right: 10; 
    }
    div.table {
      width:600px;
      margin: auto;
     padding-right: 10; 
    }
    </style>
    </head>
    <div class="jumbotron text-center">
      <h1> Machine Learning Algorithm Assessment Report </h1>
    </div>
    <div class="container">
      <div class="row">
        <div class="col-sm-4">
          <img src="1.png" alt="Smiley face" height="800" width="800">
        </div>

      </div>
    </div>
    </body>
    </html>
    """ 
    out_html.write(part_1)
    out_html.close()
# Load sets of amyloidogenic and non-amyloidogenic peptides:

def run(Fasta1, Fasta2, windows_per_frame, overlap_factor, xlabel, ylabel, pop1_label, pop2_label, htmlOutDir, htmlFname, Workdirpath):

    if not os.path.exists(htmlOutDir):
        os.makedirs(htmlOutDir)

    amyload_pos_seq = load_fasta_file(Fasta1)
    amyload_neg_seq = load_fasta_file(Fasta2)

    # Calculate quantitive features: volume and hydropathy
    mean_volume = Feature(get_aa2volume()).then(average)
    mean_hydropathy = Feature(get_aa2hydropathy()).then(average)

    fs = FeatureSet("volume'n'hydropathy")
    fs.add(mean_volume)
    fs.add(mean_hydropathy)

    amyload_pos_conv_seq = fs(amyload_pos_seq)
    amyload_neg_conv_seq = fs(amyload_neg_seq)

    # Do local Fisher:
    result = local_fisher_2d(amyload_pos_conv_seq, amyload_neg_conv_seq,
                             windows_per_frame=int(windows_per_frame), overlap_factor=int(overlap_factor))

    # Plot local Fisher:
    _plot_local_fisher_2d(result, xlabel=xlabel,
                                  ylabel=ylabel,
                                  pop1_label=pop1_label,
                                  pop2_label=pop2_label,
                                  out_file_path =os.path.join(Workdirpath, htmlOutDir, "1.png") 
                                  )

    
    #   plt.savefig(os.path.join(Workdirpath, htmlOutDir, "1.png"))

    HTML_Gen(os.path.join(Workdirpath, htmlOutDir, htmlFname))

if __name__=="__main__":
    
    
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-f1", "--Fasta1",
                        required=True,
                        default=None,
                        help="pep file")
                        
    parser.add_argument("-f2", "--Fasta2",
                        required=True,
                        default=None,
                        help="out put file name for str Descriptors")   

    parser.add_argument("-o", "--overlap_factor",
                        required=False,
                        default=5,
                        help="Path to out file")  

    parser.add_argument("-w", "--windows_per_frame",
                        required=False,
                        default=5,
                        help="Path to out file")  

    parser.add_argument("-x", "--xlabel",
                        required=True,
                        default=None,
                        help="Path to out file")  

    parser.add_argument("-y", "--ylabel",
                        required=True,
                        default=None,
                        help="Path to out file")  

    parser.add_argument("-p1", "--pop1_label",
                        required=True,
                        default=None,
                        help="Path to out file")  

    parser.add_argument("-p2", "--pop2_label",
                        required=True,
                        default=None,
                        help="Path to out file") 


    parser.add_argument("--htmlOutDir", 
                        required=False, 
                        default=os.path.join(os.getcwd(),'report_dir'), 
                        help="HTML Out Dir")

    parser.add_argument("--htmlFname", 
                        required=False, 
                        help="HTML out file", 
                        default="jai.html")

    parser.add_argument("--Workdirpath", 
                        required=False, 
                        default=os.getcwd(), 
                        help="Working Directory Path")
                                                
    args = parser.parse_args()

    run(args.Fasta1, args.Fasta2, args.windows_per_frame, args.overlap_factor, args.xlabel, args.ylabel, args.pop1_label, args.pop2_label, args.htmlOutDir, args.htmlFname, args.Workdirpath)

