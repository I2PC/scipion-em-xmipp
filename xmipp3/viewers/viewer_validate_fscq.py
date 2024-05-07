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
from pwem.viewers import (LocalResolutionViewer, EmPlotter, ChimeraView,
                          ChimeraAttributeViewer)
from pyworkflow.gui import *
from pyworkflow.protocol.params import (LabelParam, EnumParam)
from pyworkflow.viewer import ProtocolViewer, DESKTOP_TKINTER
from xmipp3.protocols.protocol_validate_fscq import (XmippProtValFit, 
                                                     RESTA_FILE_MRC, 
                                                     OUTPUT_PDBMRC_FILE, 
                                                     RESTA_FILE_NORM)

class XmippProtValFitViewer(LocalResolutionViewer, ChimeraAttributeViewer):
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
        
        group = form.addGroup('Chimera visualization')
        
        group.addParam('displayVolume', LabelParam,
                      important=True,
                      label='Display FSC-Q Volume Output')
        
        group.addParam('displayNormVolume', LabelParam,
               important=True,
               label='Display FSC-Qr Volume Output')

        group.addParam('displayPDB', EnumParam,
                      choices=['by residue', 'by atom'],
                      default=0, important=True,
                      display=EnumParam.DISPLAY_COMBO,
                      label='Display FSC-Q on PDB Output',
                      help='FSC-Q projected on the atomic model')  
        
        group.addParam('displayNormPDB', EnumParam,
              choices=['by residue', 'by atom'],
              default=0, important=True,
              display=EnumParam.DISPLAY_COMBO,
              label='Display FSC-Qr on PDB Output',
              help='FSC-Qr projected on the atomic model') 
        
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

        super()._defineParams(form)
        from pwem.wizards.wizard import ColorScaleWizardBase
        group = form.addGroup('Color settings')
        ColorScaleWizardBase.defineColorScaleParams(group, defaultLowest=-3, defaultHighest=3, defaultIntervals=21,
                                                    defaultColorMap='RdBu_r')


    def _getVisualizeDict(self):
        self.protocol._createFilenameTemplates()
        visualizeDict = {'displayVolume': self._visualize_vol,
                         'displayNormVolume': self._visualize_norm_vol,
                         'displayPDB': self._visualize_pdb,
                         'displayNormPDB': self._visualize_norm_pdb,
                         'calculateFscqNeg': self._statistics,
                         'calculateFscqPos': self._statistics}
        visualizeDict.update(ChimeraAttributeViewer._getVisualizeDict(self))
        return visualizeDict     
    
    def _create_legend(self, scale):
         
        fnCmd = self.protocol._getExtraPath("chimera_output.py")
        
        f = open(fnCmd, 'w')
        f.write("from chimerax.core.commands import run\n")
        f.write("from chimerax.graphics.windowsize import window_size\n")
        f.write("try:\n")
        f.write("    from PyQt5.QtGui import QFontMetrics\n")
        f.write("    from PyQt5.QtGui import QFont\n")
        f.write("except ModuleNotFoundError:\n")
        f.write("    from PyQt6.QtGui import QFontMetrics\n")
        f.write("    from PyQt6.QtGui import QFont\n")

        f.write("run(session, 'set bgColor white')\n")
        # get window size so we can place labels properly
        f.write("v = session.main_view\n")
        f.write("vx,vy=v.window_size\n")
        # Calculate heights and Y positions: font, scale height and firstY
        f.write('font = QFont("Arial", 12)\n')
        f.write('f = QFontMetrics(font)\n')
        f.write('_height =  1 * f.height()/vy\n') # Font height
        f.write('_half_scale_height = _height * 3.5\n') # Full height of the scale
        f.write("_firstY= 0.5 + _half_scale_height\n")  # Y location for first label
        
        val = scale
        f.write('scale = %f \n' % val)
        
        f.write("run(session, '2dlabel text -%.2f bgColor red xpos 0.01 ypos %f size 12' % (scale, _firstY)) \n")
        f.write("run(session, '2dlabel text -%.2f bgColor orange xpos 0.01 ypos %f size 12' % (scale/1.5, _firstY-_height)) \n")
        f.write("run(session, '2dlabel text -%.2f bgColor gold xpos 0.01 ypos %f size 12' % (scale/3, _firstY-2*_height)) \n")
        f.write("run(session, '2dlabel text %05.2f bgColor yellow xpos 0.01 ypos %f size 12' % (00.00, _firstY-3*_height)) \n")            
        f.write("run(session, '2dlabel text %05.2f bgColor lime xpos 0.01 ypos %f size 12' % (scale/3, _firstY-4*_height)) \n")
        f.write("run(session, '2dlabel text %05.2f bgColor cyan xpos 0.01 ypos %f size 12' % (scale/1.5, _firstY-5*_height)) \n")
        f.write("run(session, '2dlabel text %05.2f bgColor dodger blue xpos 0.01 ypos %f size 12' % (scale, _firstY-6*_height)) \n") 
        

    def _visualize_vol(self, obj, **args):
        
        self._create_legend(3)
        
        fnCmd = self.protocol._getExtraPath("chimera_output.py")
        f = open(fnCmd, 'a')
               
        f.write("run(session, 'open %s')\n" % self.protocol._getFileName(OUTPUT_PDBMRC_FILE))
        f.write("run(session, 'open %s')\n" % self.protocol._getFileName(RESTA_FILE_MRC))
        
        f.write("run(session, 'volume #2 voxelSize %s step 1')\n" % 
                self.protocol.inputVolume.get().getSamplingRate() )
        f.write("run(session, 'volume #3 voxelSize %s')\n" % 
                self.protocol.inputVolume.get().getSamplingRate() )
        f.write("run(session, 'vol #3 hide')\n")        
        f.write("run(session, 'color sample #2 map #3 palette"
                " -3.0,red:-2.0,orange:-1.0,gold:0,yellow:1.0,lime:2.0,cyan:3.0,#1e90ff')\n")
                               
        f.close()     
        view = ChimeraView(fnCmd)
        return [view]
       
    
    def _visualize_norm_vol(self, obj, **args):
        
        self._create_legend(1.5)
        
        fnCmd = self.protocol._getExtraPath("chimera_output.py")
        f = open(fnCmd, 'a')
               
        f.write("run(session, 'open %s')\n" % self.protocol._getFileName(OUTPUT_PDBMRC_FILE))
        f.write("run(session, 'open %s')\n" % self.protocol._getFileName(RESTA_FILE_NORM))
        
        f.write("run(session, 'volume #2 voxelSize %s step 1')\n" % 
                self.protocol.inputVolume.get().getSamplingRate() )
        f.write("run(session, 'volume #3 voxelSize %s')\n" % 
                self.protocol.inputVolume.get().getSamplingRate() )
        f.write("run(session, 'vol #3 hide')\n")       
        f.write("run(session, 'color sample #2 map #3 palette"
                " -1.5,red:-1.0,orange:-0.5,gold:0,yellow:0.5,lime:1.0,cyan:1.5,#1e90ff')\n")
                               
        f.close()     
        view = ChimeraView(fnCmd)
        return [view]        
    
    
    def _visualize_pdb(self, obj, **args):
        
        self._create_legend(3)
        
        fnCmd = self.protocol._getExtraPath("chimera_output.py")
        f = open(fnCmd, 'a')
               
        f.write("run(session, 'open %s')\n" % self.protocol.getFSCQFile())
         
        if self.displayPDB == self.RESIDUE:
            f.write("run(session, 'cartoon')\n")
            f.write("run(session, 'hide target ab')\n")
            f.write("run(session, 'color byattribute occupancy palette" 
                         " -3.0,red:-2.0,orange:-1.0,gold:0,yellow:1.0,lime:2.0,cyan:3.0,#1e90ff" 
                         " ave residue')\n")
                                      
        else:
            f.write("run(session, 'cartoon hide')\n")
            f.write("run(session, 'show target ab')\n")
            f.write("run(session, 'style stick')\n")
            f.write("run(session, 'color byattribute occupancy palette" 
                         " -3.0,red:-2.0,orange:-1.0,gold:0,yellow:1.0,lime:2.0,cyan:3.0,#1e90ff')\n")
                               
        f.close()     
        view = ChimeraView(fnCmd)
        return [view]
    
    
    def _visualize_norm_pdb(self, obj, **args):
        
        self._create_legend(1.5)
        
        fnCmd = self.protocol._getExtraPath("chimera_output.py")
        f = open(fnCmd, 'a')
               
        f.write("run(session, 'open %s')\n" % self.protocol.getNormFSCQFile())
         
        if self.displayNormPDB == self.RESIDUE:
            f.write("run(session, 'cartoon')\n")
            f.write("run(session, 'hide target ab')\n")
            f.write("run(session, 'color byattribute occupancy palette" 
                         " -1.5,red:-1.0,orange:-0.5,gold:0,yellow:0.5,lime:1.0,cyan:1.5,#1e90ff" 
                         " ave residue')\n")
                                      
        else:
            f.write("run(session, 'cartoon hide')\n")
            f.write("run(session, 'show target ab')\n")
            f.write("run(session, 'style stick')\n")
            f.write("run(session, 'color byattribute occupancy palette" 
                         " -1.5,red:-1.0,orange:-0.5,gold:0,yellow:0.5,lime:1.0,cyan:1.5,#1e90ff')\n")
                               
        f.close()     
        view = ChimeraView(fnCmd)
        return [view]        
         
 
    def _calculate_fscq(self, obj, **args):

        status = False
        overfittingList = []
        poorfittingList = []
        with open(os.path.abspath(self.protocol.getFSCQFile())) as f:
            linesData = f.readlines()
            for j, lin in enumerate(linesData):
                if lin.startswith('ATOM') or lin.startswith('HETATM'):
                    resnumber = int(lin[22:26])

                    if status and resnumber == resnumberCtl:
                        resname = lin[17:20].strip()
                        chain = lin[21]
                        fscq = float(lin[54:60])
                        currentFrag.append(fscq)

                    elif status and resnumber != resnumberCtl:
                        meanFscq = np.mean(currentFrag)
                        currentFrag.sort()

                        if currentFrag[0] <= -1:
                            overfittingList.append((resname, resnumberCtl, chain,
                                                     currentFrag[0], meanFscq))

                        if currentFrag[-1] >= 1:
                            poorfittingList.append((resname, resnumberCtl, chain,
                                                     currentFrag[-1],
                                                     meanFscq))
                     
                        currentFrag = []
                        resnumberCtl = resnumber
                        resname = lin[17:20].strip()
                        chain = lin[21]
                        fscq = float(lin[54:60])

                        currentFrag.append(fscq)

                    else:
                        status = True
                        currentFrag = []
                        resnumberCtl = resnumber
                        resname = lin[17:20].strip()
                        chain = lin[21]
                        fscq = float(lin[54:60])

                        currentFrag.append(fscq)
                        
        if obj == 'calculateFscqNeg':
            return overfittingList
        else:
            return poorfittingList

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
