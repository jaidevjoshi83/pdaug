# -*- coding: utf-8 -*-
"""
.. currentmodule:: modlamp.wetlab

.. moduleauthor:: modlab Alex Mueller ETH Zurich <alex.mueller@pharma.ethz.ch>

This module incorporates functions to load raw data files from wetlab experiments, calculate different
characteristics and plot.

=============================        ============================================================================
Class                                Data
=============================        ============================================================================
:py:class:`CD`                       Class for handling Circular Dichroism data.
=============================        ============================================================================
"""
import os

from os import listdir, makedirs
from os.path import join, exists, splitext

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modlamp.descriptors import GlobalDescriptor

__author__ = "Alex Müller, Gisela Gabernet"
__docformat__ = "restructuredtext en"


class CD:
    """
    Class to handle circular dichroism data files and calculate several ellipticity and helicity values.
    The class can handle data files of the *Applied Photophysics* type.
    
    For explanations of different units used in CD spectroscopy,
    visit https://www.photophysics.com/resources/7-cd-units-conversions and read the following publication:
    
    N. J. Greenfield, *Nat. Protoc.* **2006**, 1, 2876–2890.
    
    .. note::
        All files which should be read must have **4 header lines** as shown in the image below. CD data to be read
        must start in line 5 (separated in 2 columns: *Wavelength* and *Signal*).
        
    .. image:: ../docs/static/fileheader.png
        
    First line: *Molecule Name*
    
    Second line: *Sequence*
    
    Third line: *concentration in µM*
    
    Fourth line: *solvent*
    
    Recognized solvents are **W** for water and **T** for TFE.
    """
    
    def __init__(self, directory, wmin, wmax, amide=True, pathlen=1):
        """Init method for class CD.
        
        :param directory: {str} directory containing all data files to be read. Files need a **.csv** ending
        :param wmin: {int} smalles wavelength measured
        :param wmax: {int} highest wavelength measured
        :param amide: {bool} specifies whether the sequences have amidated C-termini
        :param pathlen: {float} cuvette path length in mm
        :Example:
        
        >>> cd = CD('/path/to/your/folder', 185, 260)
        >>> cd.filenames
        ['160819_Pep1_T_smooth.csv', '160819_Pep1_W_smooth.csv', '160819_Pep5_T_smooth.csv', ...]
        >>> cd.names
        ['Pep 10', 'Pep 10', 'Pep 11', 'Pep 11', ... ]
        >>> cd.conc_umol
        [33.0,  33.0,  33.0,  33.0,  33.0,  33.0,  33.0,  33.0,  33.0,  33.0,  33.0,  ... ]
        >>> cd.meanres_mw
        [114.29920769230768, 114.29920769230768, 111.68257692307689, 111.68257692307689, ... ]
         """
        
        # read filenames from directory
       # files = listdir(directory)
        files = directory.split(',')

        self.filenames = [filename for filename in files if filename.endswith('.dat')]  # get all .csv files in dir
        
        # initialize attributes
        self.names = list()
        self.sequences = list()
        self.conc_umol = list()
        self.conc_mgml = list()
        self.mw = list()
        self.meanres_mw = list()
        self.solvent = list()
        self.circular_dichroism = list()
        self.molar_ellipticity = list()
        self.meanres_ellipticity = list()
        self.helicity_values = pd.DataFrame()
        self.directory = directory
        self.wmin = wmin
        self.wmax = wmax
        self.amide = amide
        self.pathlen = pathlen
        
        self._read_header()  # call the _read_header function to fill up all attributes
    
    def _read_header(self):
        """Priveat method called by ``__init__`` to read all file headers into the class attributes and calculate
        sequence dependant values.
        
        :return: headers in class attributes.
        """
        
        d = GlobalDescriptor('X')  # template
        
        # loop through all files in the directory
        for i, file in enumerate(self.filenames):
            with open(join(file)) as f:  # read first 4 lines as header, rest as data
                head = [next(f) for _ in range(4)]
                data = [next(f) for _ in range(4, (int(self.wmax) - int(self.wmin)) + 5)]
            
            # read headers into class attributes
            name = head[0].split('\r\n')[0]
            self.names.append(name)
            sequence = head[1].split('\r\n')[0].strip()
            self.sequences.append(sequence)
            umol = float(head[2].split('\r\n')[0])
            self.conc_umol.append(umol)
            self.solvent.append(head[3].split('\r\n')[0])
            
            # read CD data
            wlengths = [int(line.split('\t')[0]) for line in data]  # get rid of s***** line ends
            ellipts = [float(line.split('\t')[1].split('\r\n')[0]) for line in data]
            self.circular_dichroism.append(np.array(list(zip(wlengths, ellipts))))
            
            # calculate MW and transform concentration to mg/ml
            d.sequences = [sequence]
            d.calculate_MW(amide=self.amide)
            self.mw.append(d.descriptor[0][0])
            self.conc_mgml.append(self.mw[i] * umol / 10 ** 6)
            self.meanres_mw.append(self.mw[i] / (len(sequence) - 1))  # mean residue molecular weight (MW / n-1)
    
    def calc_molar_ellipticity(self):
        """Method to calculate the molar ellipticity for all loaded data in :py:attr:`circular_dichroism` according
        to the following formula:
        
        .. math::
            
            [\Theta] = (\Theta * MW) / (c * l)

        :return: {numpy array} molar ellipticity in :py:attr:`molar_ellipticity`
        :Example:
        
        >>> cd.calc_molar_ellipticity()
        >>> cd.molar_ellipticity
        array([[  260.,  -1.40893636e+04],
               [  259.,  -1.00558182e+04],
               [  258.,  -1.25173636e+04], ...
        """
        
        for i, data in enumerate(self.circular_dichroism):
            # calculate molar ellipticity: (theta * MW) / (conc * pathlength); and concat. with wavelengths
            mol_ellipt = (data[:, 1] * self.mw[i]) / (self.conc_mgml[i] * self.pathlen)
            self.molar_ellipticity.append(np.array(list(zip(data[:, 0], mol_ellipt))))

        return self.molar_ellipticity
  
    
    def calc_meanres_ellipticity(self):
        """Method to calculate the mean residue ellipticity for all loaded data in :py:attr:`circular_dichroism`
        according to the following formula:
        
        .. math::
            
            (\Theta * MRW) / (c * l) = [\Theta]
            
            MRW = MW / (n - 1)
        
        With *MRW* = mean residue weight (g/mol), *n* = number of residues in the peptide, *c* = concentration (mg/mL)
        and *l* = cuvette path length (mm).
        
        :return: {numpy array} molar ellipticity in :py:attr:`meanres_ellipticity`
        :Example:
        
        >>> cd.calc_meanres_ellipticity()
        >>> cd.meanres_ellipticity
        array([[   260.        ,   -2669.5804196],
               [   259.        ,   -3381.3286713],
               [   258.        ,   -3872.5174825], ...
        """
        
        for i, data in enumerate(self.circular_dichroism):
            # calculate molar ellipticity: (theta * mrw) / (conc * pathlength); and concat. with wavelengths
            mol_ellipt = (data[:, 1] * self.meanres_mw[i]) / (self.conc_mgml[i] * self.pathlen)
            self.meanres_ellipticity.append(np.array(list(zip(data[:, 0], mol_ellipt))))

        return self.meanres_ellipticity
   
    
    def _plot_single(self, w, d, col, y_label, title, filename, y_min, y_max, OutPath):
        """Private plot function used by :py:func:`modlamp.wetlab.CD.plot()` for plotting single CD plots"""
        
        fig, ax = plt.subplots()
        line = ax.plot(w, d / 1000.)  # used legend is 10^3 so divide by 1000
        plt.setp(line, color=col, linewidth=2.0)
        plt.title(title, fontsize=18, fontweight='bold')
        ax.set_xlabel('Wavelength [nm]', fontsize=16)
        ax.set_ylabel(y_label, fontsize=16)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        plt.xlim(np.min(w), np.max(w))
        plt.ylim((y_min / 1000., y_max / 1000.))
        img_name = splitext(filename)[0] + '.pdf'
        #plt.savefig(join(self.directory, 'PDF', img_name), dpi=150)
        #print os.path.join(OutPath, "2.png")
        plt.savefig(os.path.join(OutPath, "1.png"))
    
    def _plot_double(self, w, dt, dw, y_label, title, filename, y_min, y_max, OutPath):
        """Private plot function used by :py:func:`modlamp.wetlab.CD.plot()` for plotting combined CD plots"""
        
        fig, ax = plt.subplots()
        line1 = ax.plot(w, dt / 1000.)
        line2 = ax.plot(w, dw / 1000.)
        plt.setp(line1, color='r', linewidth=2.0, label='TFE', linestyle='--')
        plt.setp(line2, color='b', linewidth=2.0, label='Water')
        plt.title(title, fontsize=18, fontweight='bold')
        ax.set_xlabel('Wavelength [nm]', fontsize=16)
        ax.set_ylabel(y_label, fontsize=16)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        plt.xlim(np.min(w), np.max(w))
        plt.ylim((y_min / 1000., y_max / 1000.))
        plt.legend(loc=1)
        img_name = splitext(filename)[0] + '_M.pdf'
        #plt.savefig(join(self.directory, 'PDF', img_name), dpi=150)

        print os.path.join(OutPath, "2.png")
        plt.savefig(os.path.join(OutPath, "2.png"))
    
    def _plot_all(self, data, w, y_lim, OutPath):

        if not os.path.exists(self.htmlOutDir):
            os.makedirs(self.htmlOutDir)
        """Private plot function used by :py:func:`modlamp.wetlab.CD.plot()` for plotting combined CD plots"""
        
        colors = ['#53777A', '#542437', '#C02942', '#D95B43', '#ECD078', '#CFF09E', '#A8DBA8', '#79BD9A', '#3B8686',
                  '#0B486B', '#2790B0', '#94BA65', '#353432', '#4E4D4A', '##808080', '#CCCCCC']
        
        fig, ax = plt.subplots()
        y_label = ''  # assign empty
        y_min, y_max = (0, 1)  # assign empty
        
        for i, f in enumerate(self.filenames):
            d, _, y_label, y_min, y_max = self._check_datatype(data, i, 'all')
            
            if y_lim:
                y_min = 1000 * y_lim[0]  # * 1000 because axis are usually shown as 10^3
                y_max = 1000 * y_lim[1]
            
            vars()['line' + str(i)] = ax.plot(w, d / 1000.)  # mark the line plots with the iterator, for labelling
            

            try:
                plt.setp(vars()['line' + str(i)], color=colors[i], linewidth=1.5, label='%s' % f.split('.')[0], linestyle='-')

            except IndexError:  # if more data than colors: start with dashed lines
                plt.setp(vars()['line' + str(i)], color=colors[i - len(colors)], linewidth=1.5, label='%s' % f.split('.')[0], linestyle='--')
        
        plt.title("Combined Plot", fontsize=18, fontweight='bold')
        ax.set_xlabel('Wavelength [nm]', fontsize=16)
        ax.set_ylabel(y_label, fontsize=16)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        plt.xlim(np.min(w), np.max(w))
        plt.ylim((y_min / 1000., y_max / 1000.))
        plt.legend(loc=1)
        #plt.savefig(join(self.directory, 'PDF', 'all.pdf'), dpi=150)
        #print "ok"
        #plt.show()
        #print os.path.join(OutPath, "2.png")
        plt.savefig(os.path.join(OutPath, "3.png"))
    
    def _check_datatype(self, data, i, comb_flag):
        """Private function to check data type; used by :py:func`modlamp.wetlab.CD.plot` and
        :py:func`modlamp.wetlab.CD.dichroweb`"""
        
        d2 = []
        if data == 'mean_residue_ellipticity':
            d = self.meanres_ellipticity[i][:, 1]
            if comb_flag == 'solvent' and i % 2 == 0:
                d2 = self.meanres_ellipticity[i + 1][:, 1]
            y_label = r"$[\Theta] \ast 10^{-3} (deg \ast cm^2 \ast dmol^{-1})$"
            y_min = np.min(d) * 1.1
            y_max = np.max(d) * 1.1
        elif data == 'molar_ellipticity':
            d = self.molar_ellipticity[i][:, 1]
            if comb_flag == 'solvent' and i % 2 == 0:
                d2 = self.molar_ellipticity[i + 1][:, 1]
            y_label = r"$[\Theta] \ast 10^{-3} (deg \ast cm^2 \ast dmol^{-1})$"
            y_min = np.min(d) * 1.1
            y_max = np.max(d) * 1.1
        else:
            d = self.circular_dichroism[i][:, 1]
            if comb_flag == 'solvent' and i % 2 == 0:
                d2 = self.molar_ellipticity[i + 1][:, 1]
            y_label = r"$\Delta A \ast 32.986 \ast 10^{-3}$"
            y_min = np.min(d) * 1.1
            y_max = np.max(d) * 1.1
        
        return d, d2, y_label, y_min, y_max
    
    def plot(self, Workdirpath, htmlOutDir, data='mean_residue_ellipticity', combine='solvent', ylim=None):
        if not os.path.exists(os.path.join(Workdirpath, htmlOutDir)):
            os.makedirs(os.path.join(Workdirpath, htmlOutDir))
        """Method to generate CD plots for all read data in the initial directory.
        
        :param data: {str} which data should be plotted (``mean residue ellipticity``, ``molar ellipticity`` or
            ``circular dichroism``)
        :param combine: {str} if ``solvent``, overlays of different solvents will be created for the same molecule.
            The amino acid sequence in the header is used to find corresponding data.
            if ``all``, all data is combined in one single plot. To ignore combination, pass an empty string.
        :param ylim: {tuple} If not none, this tuple of values is taken as the minimum and maximum of the y axis
        :return: .pdf plots saved to the directory containing the read files.
        :Example:
        
        >>> cd = CD('/path/to/your/folder', 185, 260)
        >>> cd.calc_meanres_ellipticity()
        >>> cd.plot(data='mean residue ellipticity', combine='solvent')
        
        .. image:: ../docs/static/cd1.png
            :height: 300px
        .. image:: ../docs/static/cd2.png
            :height: 300px
        .. image:: ../docs/static/cd3.png
            :height: 300px
        """

        # prepare combination of solvent plots
        if combine == 'solvent':
            d = {s: self.sequences.count(s) for s in set(self.sequences)}  # create dict with seq counts for combine
            if sum(d.values()) != 2 * len(d.values()):
                raise ValueError
        # check if output folder exists already, else create one
        #if not exists(join(self.directory, 'PDF')):
            #makedirs(join(self.directory, 'PDF'))
        
        w = range(int(self.wmax), int(self.wmin) - 1, -1)  # wavelengths
        
        # check input data option
        if data in ['mean_residue_ellipticity', 'molar_ellipticity', 'circular_dichroism']:
            # loop through all data for single plots

            for i, f in enumerate(self.filenames):
                
                # get data type to be plotted
                d, d2, y_label, y_min, y_max = self._check_datatype(data, i, combine)
                
                if self.solvent[i] == 'T':  # color
                    col = 'r'
                else:
                    col = 'b'
                
                if ylim:
                    y_min = 1000 * ylim[0]  # * 1000 because axis are usually shown as 10^3
                    y_max = 1000 * ylim[1]
                
                # plot single plots
                self._plot_single(w, d, col, y_label, self.names[i] + ' ' + self.solvent[i], f, y_min, y_max, os.path.join(Workdirpath, htmlOutDir))
                # plot mixed plots
                if combine == 'solvent' and i % 2 == 0:
                    self._plot_double(w, d, d2, y_label, self.names[i], f, y_min, y_max, os.path.join(Workdirpath, htmlOutDir))
            
            if combine == 'all':
                self._plot_all(data, w, ylim, os.path.join(Workdirpath, htmlOutDir))

        else:
            print("ERROR\nWrong data option given!\nAvailable:")
            print("['mean_residue_ellipticity', 'molar_ellipticity', 'circular_dichroism']")
        

    
    def dichroweb(self, data='mean_residue_ellipticity'):
        """Method to save the calculated CD data into DichroWeb readable format (semi-colon separated). The produced
        files can then directly be uploaded to the `DichroWeb <http://dichroweb.cryst.bbk.ac.uk>`_ analysis tool.
        
        :param data: {str} which data should be plotted (``mean residue ellipticity``, ``molar ellipticity`` or
            ``circular dichroism``)
        :return: .csv data files saved to the directory containing the read files.
        """
        # check if output folder exists already, else create one

        if not exists(os.path.join(os.getcwd(), 'Dichro')):
            makedirs(os.path.join(os.getcwd(), 'Dichro'))
        
        if data in ['mean_residue_ellipticity', 'molar_ellipticity', 'circular_dichroism']:
            # loop through all data for single plots
            for i, f in enumerate(self.filenames):
               
                # get data type to be plotted
                d, _, _, _, _ = self._check_datatype(data, i, False)
                w = range(int(self.wmax), int(self.wmin) - 1, -1)  # wavelengths
                dichro = pd.DataFrame(data=zip(w, d), columns=["V1", "V2"], dtype='float')
                fname = f.split('/')[-1].split('.')[0] + '.tsv'
                dichro.to_csv(os.path.join(os.getcwd(),'Dichro', fname), sep=';', index=False)
    
    def helicity(self, temperature=24., k=3.5, induction=True, filename=None):
        """Method to calculate the percentage of helicity out of the mean residue ellipticity data.
        The calculation is based on the fromula by Fairlie and co-workers:
        
        .. math::
            [\Theta]_{222\infty} = (-44000 * 250 * T) * (1 - k / N)
        
        The helicity is then calculated as the ratio of
        
        .. math::
            ([\Theta]_{222} / [\Theta]_{222\infty}) * 100 \%
        
        :Reference: `Shepherd, N. E., Hoang, H. N., Abbenante, G. & Fairlie, D. P. J. Am. Chem. Soc. 127, 2974–2983
            (2005). <https://dx.doi.org/10.1021/ja0456003>`_
        :param temperature: {float} experiment temperature in °C
        :param k: {float, 2.4 - 4.5} finite length correction factor. Can be adapted to the helicity of a known peptide.
        :param induction: {bool} wether the helical induction upon changing from one solvent to another should be
            calculated.
        :param filename: {str} if given, helicity data is saved to the file "filename".csv
        :return: approximate helicity for every sequence in the attribute :py:attr:`helicity_values`.
        :Example:
        
        >>> cd.calc_meanres_ellipticity()
        >>> cd.helicity(temperature=24., k=3.492185008, induction=True)
        >>> cd.helicity_values
                    Name    Solvent   Helicity  Induction
            0  Aurein2.2d2       T    100.0     3.823
            1  Aurein2.2d2       W    26.16     0.000
            2       Klak14       T    76.38     3.048
            3       Klak14       W    25.06     0.000
        """
        
        values = self.meanres_ellipticity
        if values:
            hel = []
            for i, v in enumerate(values):
                indx = np.where(v[:, 0] == 222.)[0][0]  # get index of wavelength 222 nm
                hel_100 = (-44000. + 250. * temperature) * (1. - (float(k) / len(self.sequences[i])))  # inf hel 222
                hel.append(round((v[indx, 1] / hel_100) * 100., 2))
            
            self.helicity_values = pd.DataFrame(np.array([self.names, self.solvent, hel]).T, columns=['Name', 'Solvent',
                                                                                                      'Helicity'])
            if induction:
                induct = []
                try:
                    for i in self.helicity_values.index:
                        if self.helicity_values.iloc[i]['Name'] == self.helicity_values.iloc[i + 1]['Name'] and \
                                        self.helicity_values.iloc[i]['Solvent'] != self.helicity_values.iloc[i + 1][
                                    'Solvent']:
                            induct.append(round(float(self.helicity_values.iloc[i]['Helicity']) / float(
                                    self.helicity_values.iloc[i + 1]['Helicity']),
                                                3))  # if following entry is same molecule
                            # but not same solvent, calculate the helical induction and round to .3f
                        else:  # else just append 0
                            induct.append(0.)
                
                except IndexError:  # at the end of the DataFrame, an index error will be raised because of i+1
                    induct.append(0.)
                    self.helicity_values['Induction'] = induct
            
            if filename:
                self.helicity_values.to_csv(filename, index=False, sep="\t")
        
        else:
            print("ERROR\nmeanres_ellipticity data array empty, call the calculate function first:")
            print("calc_meanres_ellipticity()")


if __name__=="__main__":
    


    import os, glob
    import argparse
    import string


    parser = argparse.ArgumentParser()

    parser.add_argument("-I", "--InFiles", required=True, default=None, help="Path to target tsv file")
    parser.add_argument("-M", "--Method", required=True, default=None, help="mean_residue_ellipticity, molar_ellipticity or circular dichroism" )
    parser.add_argument("-Wn", "--Wmin", required=True, default=None, help="Method")
    parser.add_argument("-Wx", "--Wmax", required=True, default=None, help="Roatate ticks")
    parser.add_argument("-a", "--amide", required=True, default=None, help="Roatate ticks")
    parser.add_argument("-p", "--pathlen", required=False,  default=1,  help="Figure Hight")
    parser.add_argument("-L", "--CalcType", required=False,  default='molar_ellipticity',  help="Figure Hight")
    parser.add_argument("--htmlOutDir",  required=False, default=os.path.join(os.getcwd(),'report_dir'), help="HTML Out Dir")
    parser.add_argument("--htmlFname", required=False, help="HTML out file", default="Out.html")
    parser.add_argument("--Workdirpath", required=False, default=os.getcwd(), help="Working Directory Path")
    parser.add_argument("-C","--Combined", required=False, default='solvent')
    parser.add_argument("-O","--OutFile", required=False, default='Out.csv', help="Out.tsv")
    parser.add_argument("-T", "--Temperature", required=False, default=24, help="Method")
    parser.add_argument("-K", "--FinLenCorrOption", required=False, default=3.0, help="")
    parser.add_argument("-D", "--Induction", required=False, default=True, help="")
    parser.add_argument("-H", "--HelicityOut", required=False, default='Helicity.tsv', help="")

    args = parser.parse_args()


    def HTML_Gen(html, ImgPath):

        Im = glob.glob(os.path.join(ImgPath,'*.png'))

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
            <div class="col-sm-6">
        """
        out_html.write(part_1)

        for I in Im:

            Img = "     <img src=%s alt='Smiley face' height='500' width='700'>"%I.split('/')[-1]
            out_html.write(Img+'\n')

        part_2 = """
            </div>
          </div>
        </div>
        </body>
        </html>
        """ 
        out_html.write(part_2)
        out_html.close()


    if args.amide == 'true':
        Am = True 
    elif args.amide == 'false':
        Am = False
    else:
        pass

    if args.Induction == 'true':
        Ind  = True 
    elif args.Induction == 'false':
        Ind = False
    else:
        pass

    cd = CD(args.InFiles, int(args.Wmin), int(args.Wmax), amide=Am, pathlen=float(args.pathlen))


    if args.Method == "Plot_data":

        if args.CalcType == 'molar_ellipticity':
            cd.calc_molar_ellipticity()
            cd.dichroweb(args.CalcType)
            cd.plot(Workdirpath=args.Workdirpath, htmlOutDir=args.htmlOutDir, data=args.CalcType, combine='solvent')
        elif args.CalcType == 'mean_residue_ellipticity':
            cd.calc_meanres_ellipticity()
            cd.dichroweb(args.CalcType)
            cd.plot(Workdirpath=args.Workdirpath, htmlOutDir=args.htmlOutDir, data=args.CalcType, combine='solvent')
        if args.CalcType == 'circular_dichroism':
            cd.calc_molar_ellipticity()
            cd.dichroweb(args.CalcType)
            cd.plot(Workdirpath=args.Workdirpath, htmlOutDir=args.htmlOutDir, data=args.CalcType, combine='solvent')
        HTML_Gen(os.path.join(args.Workdirpath, args.htmlOutDir, args.htmlFname), os.path.join(args.Workdirpath, args.htmlOutDir),)

    elif args.Method == "Calculate_Halicity":

        cd.calc_meanres_ellipticity()
        cd.helicity(temperature=float(args.Temperature), k=float(args.FinLenCorrOption), induction=Ind, filename=args.HelicityOut)
        cd.helicity_values
    else:
        pass

