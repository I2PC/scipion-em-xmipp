# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     Erney Ramirez Aportela (eramirez@cnb.csic.es)
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************

import os
from tkinter import Tk, CENTER, Scrollbar, TOP, BOTH, Y

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
from pyworkflow.utils import getExt, removeExt
from os.path import abspath
from pwem.viewers.viewer_chimera import (Chimera)
from pyworkflow.gui import *
from pyworkflow.protocol.params import (LabelParam, EnumParam)
from pyworkflow.viewer import ProtocolViewer, DESKTOP_TKINTER
from xmipp3.protocols.protocol_validate_fscq import (XmippProtValFit, 
                                                     RESTA_FILE_MRC, 
                                                     OUTPUT_PDBMRC_FILE, 
                                                     PDB_VALUE_FILE)


class XmippProtValFitViewer(ProtocolViewer):
    """
    Visualization tools for validation fsc-q.
    
    FSC-Q is a Xmipp package for evaluate the map-to-model fit
    """
    _label = 'viewer validation_fsc-q'
    _targets = [XmippProtValFit]      
    _environments = [DESKTOP_TKINTER]
    
    RESIDUE = 0
    ATOM = 1
    
    def __init__(self, *args, **kwargs):
        ProtocolViewer.__init__(self, *args, **kwargs)
    
    def _defineParams(self, form):
        self._env = os.environ.copy()
        form.addSection(label='FSC-Q results')
        
        group = form.addGroup('Visualization in Chimera')
        
        group.addParam('displayVolume', LabelParam,
                      important=True,
                      label='Display Volume Output')


        group.addParam('displayPDB', EnumParam,
                      choices=['by residue', 'by atom'],
                      default=0, important=True,
                      display=EnumParam.DISPLAY_COMBO,
                      label='Display PDB Output')  
        
        group = form.addGroup('Statistics')
        
        group.addParam('calculateFscqNeg', LabelParam,
                      important=True,
                      label='Amino acids with possible overfitting',   
                      help='Amino acids that have atoms with FSC-Q < -1'
                      ' are determined. It is suggested that these amino acids '
                      ' be re-checked in order to improve the fit.') 
        group.addParam('calculateFscqPos', LabelParam,
                      important=True,
                      label='Amino acids with low resolvability',   
                      help='Amino acids that have atoms with FSC-Q > 1'
                      ' are determined. It is suggested that these amino acids '
                      ' be re-checked in order to improve the fit.')       
            
        
    def _getVisualizeDict(self):
        self.protocol._createFilenameTemplates()
        visualizeDict = {'displayVolume': self._visualize_vol,
                         'displayPDB': self._visualize_pdb,
                         'calculateFscqNeg': self._statistics,
                         'calculateFscqPos': self._statistics}
        return visualizeDict    

    def _visualize_vol(self, obj, **args):
         

        fnRoot = os.path.abspath(self.protocol._getExtraPath())
        
        _inputVol = self.protocol.inputVolume.get()
        fnCmd = self.protocol._getTmpPath("chimera_VOLoutput.cmd")
        
        f = open(fnCmd, 'w')

        f.write("open %s\n" % (fnRoot+'/'+OUTPUT_PDBMRC_FILE))
        f.write("open %s\n" % (fnRoot+'/'+RESTA_FILE_MRC))           
        f.write("volume #0 voxelSize %f step 1\n" % (_inputVol.getSamplingRate()))
        f.write("volume #1 voxelSize %f\n" % (_inputVol.getSamplingRate()))
        f.write("vol #1 hide\n")
        f.write("scolor #0 volume #1 perPixel false cmap -3,#ff0000:"
                "0,#ffff00:1,#00ff00:2,#00ffff:3,#0000ff\n")
        f.write("colorkey 0.01,0.05 0.02,0.95 -3 #ff0000 -2 #ff4500 -1 #ff7f00 "
                 "0 #ffff00 1  #00ff00 2 #00ffff 3 #0000ff\n")        

        f.close()

        # run in the background
        Chimera.runProgram(Chimera.getProgram(), fnCmd+"&")
        return []
    
    
    def _visualize_pdb(self, obj, **args):
        
        # show coordinate axis
        fnRoot = os.path.abspath(self.protocol._getExtraPath())
        fnCmd = self.protocol._getTmpPath("chimera_PDBoutput.cmd")
        f = open(fnCmd, 'w')
        #open PDB

        if self.displayPDB == self.RESIDUE:
            f.write("open %s\n" % (fnRoot+'/'+PDB_VALUE_FILE))
            f.write("rangecol occupancy,r -3 red 0 white 1 green 2 cyan 3 blue\n")
        else:
            f.write("open %s\n" % (fnRoot+'/'+PDB_VALUE_FILE))
            f.write("display\n")
            f.write("~ribbon\n")
            f.write("rangecol occupancy,a -3 red 0 white 1 green 2 cyan 3 blue\n")  
        f.write("colorkey 0.01,0.05 0.02,0.95 -3 #ff0000 -2 #ff4500 -1 #ff7f00 "  
                "0 white 1  #00ff00 2 #00ffff 3 #0000ff\n")    
        f.close()  
                     
        Chimera.runProgram(Chimera.getProgram(), fnCmd+"&")
        return []

    def _calculate_fscq(self, obj, **args):

        fnRoot = os.path.abspath(self.protocol._getExtraPath())
        bool = 0
        overfitting_list = []
        poorfitting_list = []
        with open(fnRoot + '/' + PDB_VALUE_FILE) as f:

            lines_data = f.readlines()
            for j, lin in enumerate(lines_data):
                if lin.startswith('ATOM') or lin.startswith('HETATM'):
                    resnumber = int(lin[22:26])

                    if bool == 1 and resnumber == resnumber_ctl:
                        resatomname = lin[12:16].strip()
                        resname = lin[17:20].strip()
                        chain = lin[21]
                        fscq = float(lin[54:60])
                        current_frag.append(fscq)

                    elif bool == 1 and resnumber != resnumber_ctl:
                        meanFscq = np.mean(current_frag)
                        current_frag.sort()

                        if current_frag[0] <= -1:
                            overfitting_list.append((resname, resnumber_ctl, chain,
                                                     current_frag[0], meanFscq))

                        if current_frag[-1] >= 1:
                            poorfitting_list.append((resname, resnumber_ctl, chain,
                                                     current_frag[-1],
                                                     meanFscq))
                     
                        current_frag = []
                        resnumber_ctl = resnumber
                        resatomname = lin[12:16].strip()
                        resname = lin[17:20].strip()
                        chain = lin[21]
                        fscq = float(lin[54:60])

                        current_frag.append(fscq)

                    else:
                        bool = 1
                        current_frag = []
                        lines_aa = []
                        resnumber_ctl = resnumber
                        resatomname = lin[12:16].strip()
                        resname = lin[17:20].strip()
                        chain = lin[21]
                        fscq = float(lin[54:60])

                        current_frag.append(fscq)
                        
        if obj == 'calculateFscqNeg':
            return overfitting_list
        else:
            return poorfitting_list

    def _statistics(self, obj, **args):
        mainFrame = Tk()
        statistics = Statistics('FSC-Q Statistics', mainFrame,
                   statistics=self._calculate_fscq(obj, **args))
        statistics.mainloop()


class Statistics(ttk.Frame):
    """
        Windows to hold a plugin manager help
        """
    def __init__(self, title, mainFrame, **kwargs):
        super().__init__(mainFrame)
        mainFrame.title(title)
        mainFrame.configure(width=1500, height=400)
        self.FrameTable = tk.PanedWindow(mainFrame, orient=tk.VERTICAL)
        self.FrameTable.pack(side=TOP, fill=BOTH, expand=Y)
        self.aminolist = kwargs['statistics']
        self.fill_statistics()
        self.FrameTable.rowconfigure(0, weight=1)
        self.FrameTable.columnconfigure(0, weight=1)
        self.flag=False

    def fill_statistics(self):
        self.addStatisticsTable(0, 0)

    def addStatisticsTable(self, row, column):
        """
        This methods shows some statistics values
        """
        def fill_table():
            try:
                for values in self.aminolist:

                    if self.aminolist.index(values) % 2 == 0:
                        self.Table.insert('', tk.END, text='',
                                     values=(values[0] + str(values[1]) + "-" + values[2],
                                             round(values[3], 2),
                                             round(values[4], 2)),
                                     tags=('even',))
                    else:
                        self.Table.insert('', tk.END, text='',
                                     values=(values[0] + str(values[1]) + "-" + values[2],
                                             round(values[3], 2),
                                             round(values[4], 2)),
                                     tags=('odd',))

            except Exception as e:
                pass
            
        self.columns = ("aminoacid", "fscq", "mean")
        self.columsText = ("Aminoacids", "atom FSC-Q", "Mean FSC-Q aminoacid")

        self.Table = ttk.Treeview(self.FrameTable, columns=self.columns)
        self.Table.grid(row=row, column=column, sticky='news')
        self.Table.tag_configure("heading", background='sky blue', foreground='black',
                            font=('Calibri', 10, 'bold'))
        self.Table.tag_configure('even', background='white', foreground='black')
        self.Table.tag_configure('odd', background='gainsboro', foreground='black')
        self.Table.heading(self.columns[0], text=self.columsText[0])
        self.Table.heading(self.columns[1], text=self.columsText[1])
        self.Table.heading(self.columns[2], text=self.columsText[2])
        self.Table.column("#0", width=0, minwidth=0, stretch=False)
        self.Table.column(self.columns[0], anchor=CENTER)
        self.Table.column(self.columns[1], anchor=CENTER)
        self.Table.column(self.columns[2], anchor=CENTER)
        yscroll = Scrollbar(self.FrameTable, orient='vertical', command=self.Table.yview)
        yscroll.grid(row=row, column=column + 1, sticky='news')
        self.Table.configure(yscrollcommand=yscroll.set)
        yscroll.configure(command=self.Table.yview)
        self.Table.bind("<Button-1>", self._orderTable, True)

        fill_table()

    def _orderTable(self, event):
        x, y, widget = event.x, event.y, event.widget
        column = self.Table.identify_column(x)
        row = self.Table.identify_column(y)
        if row == '#1':  # click over heading
            col = 0
            if column == '#2':
                col = 1
            elif column == '#3':
                col = 2
            self.Table.heading(self.columns[col], text=self.columsText[col], command=lambda: \
                            self.treeview_sort_column(col, self.flag))
    
    def treeview_sort_column(self, col, reverse):
        if col == 0:
            l = [(self.Table.set(k, col), k) for k in self.Table.get_children('')]
        else:
            l = [(float(self.Table.set(k, col)), k) for k in
                 self.Table.get_children('')]
        l.sort(reverse=reverse)

        for index, (_, k) in enumerate(l):
            self.Table.move(k, '', index)

        self.Table.heading(col,
            command=lambda: self.treeview_sort_column(col, not reverse))
        self.flag = not self.flag