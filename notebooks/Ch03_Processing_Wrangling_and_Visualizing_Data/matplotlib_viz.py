# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 09:56:39 2017

@author: Raghav Bali
"""

"""

This script visualizes data using matplotlib  

``Execute``
        $ python matplotlib_viz.py

"""

import numpy as np
import matplotlib.pyplot as plt


if __name__=='__main__':
    
    # sample plot
    x = np.linspace(-10, 10, 50)
    y=np.sin(x)
    
    plt.plot(x,y)
    plt.title('Sine Curve using matplotlib')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.show()
    
    
    # figure
    plt.figure(1)
    plt.plot(x,y)
    plt.title('Fig1: Sine Curve')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.show()
    
    plt.figure(2)
    y=np.cos(x)
    plt.plot(x,y)
    plt.title('Fig2: Cosine Curve')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.show()
    
    ### subplot
    
    # fig.add_subplot
    y = np.sin(x)
    figure_obj = plt.figure()
    ax1 = figure_obj.add_subplot(2,2,1)
    ax1.plot(x,y)

    ax2 = figure_obj.add_subplot(2,2,2)
    ax3 = figure_obj.add_subplot(2,2,3)
    
    ax4 = figure_obj.add_subplot(2,2,4)
    ax4.plot(x+10,y)
    plt.show()
   
    
    # plt.subplots
    fig, ax_list = plt.subplots(2,1,sharex=True)
    y= np.sin(x)
    ax_list[0].plot(x,y)
    
    y= np.cos(x)
    ax_list[1].plot(x,y)
    plt.show()
    
    
    # plt.subplot (creates figure and axes objects automatically)
    plt.subplot(2,2,1)
    y = np.sin(x)    
    plt.plot(x,y)

    plt.subplot(2,2,2)
    y = np.cos(x)
    plt.plot(x,y)

    plt.subplot(2,1,2)
    y = np.tan(x)
    plt.plot(x,y)  
    
    plt.show()
    
    
    # subplot2grid
    y = np.abs(x)
    z = x**2
    
    plt.subplot2grid((4,3), (0, 0), rowspan=4, colspan=2)
    plt.plot(x, y,'b',x,z,'r')
    
    ax2 = plt.subplot2grid((4,3), (0, 2),rowspan=2)
    plt.plot(x, y,'b')
    plt.setp(ax2.get_xticklabels(), visible=False)

    plt.subplot2grid((4,3), (2, 2), rowspan=2)
    plt.plot(x, z,'r')
    
    plt.show()
    
      
    ### formatting
    
    y = x
    
    # color
    ax1 = plt.subplot(611)
    plt.plot(x,y,color='green')
    ax1.set_title('Line Color')
    plt.setp(ax1.get_xticklabels(), visible=False)
    
    # linestyle
    # linestyles -> '-','--','-.', ':', 'steps'
    ax2 = plt.subplot(612,sharex=ax1)
    plt.plot(x,y,linestyle='--')
    ax2.set_title('Line Style')
    plt.setp(ax2.get_xticklabels(), visible=False)
    
    # marker
    # markers -> '+', 'o', '*', 's', ',', '.', etc
    ax3 = plt.subplot(613,sharex=ax1)
    plt.plot(x,y,marker='*')
    ax3.set_title('Point Marker')
    plt.setp(ax3.get_xticklabels(), visible=False)
    
    # line width
    ax4 = plt.subplot(614,sharex=ax1)
    line = plt.plot(x,y)
    line[0].set_linewidth(3.0)
    ax4.set_title('Line Width')
    plt.setp(ax4.get_xticklabels(), visible=False)
    
    # alpha
    ax5 = plt.subplot(615,sharex=ax1)
    alpha = plt.plot(x,y)
    alpha[0].set_alpha(0.3)
    ax5.set_title('Line Alpha')
    plt.setp(ax5.get_xticklabels(), visible=False)
    
    # combine linestyle
    ax6 = plt.subplot(616,sharex=ax1)
    plt.plot(x,y,'b^')
    ax6.set_title('Styling Shorthand')
    
    fig = plt.gcf()
    fig.set_figheight(15)
    plt.show()
    
    
    # legends
    y = x**2
    z = x
    
    plt.plot(x,y,'g',label='y=x^2')
    plt.plot(x,z,'b:',label='y=x')
    plt.legend(loc="best")
    plt.title('Legend Sample')
    plt.show()
    
    # legend with latex formatting
    plt.plot(x,y,'g',label='$y = x^2$')
    plt.plot(x,z,'b:',linewidth=3,label='$y = x^2$')
    plt.legend(loc="best",fontsize='x-large')
    plt.title('Legend with LaTEX formatting')
    plt.show()
    
    
    ## axis controls
    # secondary y-axis
    fig, ax1 = plt.subplots()
    ax1.plot(x,y,'g')
    ax1.set_ylabel(r"primary y-axis", color="green")
    
    ax2 = ax1.twinx()
    
    ax2.plot(x,z,'b:',linewidth=3)
    ax2.set_ylabel(r"secondary y-axis", color="blue")
    
    plt.title('Secondary Y Axis')
    plt.show()    
    
    # ticks
    y = np.log(x)
    z = np.log2(x)
    w = np.log10(x)
    
    plt.plot(x,y,'r',x,z,'g',x,w,'b')
    plt.title('Default Axis Ticks') 
    plt.show()       
    
    # axis-controls
    plt.plot(x,y,'r',x,z,'g',x,w,'b')
    # values: tight, scaled, equal,auto
    plt.axis('tight')
    plt.title('Tight Axis') 
    plt.show()

    # manual
    plt.plot(x,y,'r',x,z,'g',x,w,'b')
    plt.axis([0,2,-1,2])
    plt.title('Manual Axis Range') 
    plt.show()       
        
    # Manual ticks      
    plt.plot(x, y)
    ax = plt.gca()
    ax.xaxis.set_ticks(np.arange(-2, 2, 1))
    plt.grid(True)
    plt.title("Manual ticks on the x-axis")
    plt.show()
    
    
    # minor ticks
    plt.plot(x, z)
    plt.minorticks_on()
    ax = plt.gca()
    ax.yaxis.set_ticks(np.arange(0, 5))
    ax.yaxis.set_ticklabels(["min", 2, 4, "max"])
    plt.title("Minor ticks on the y-axis")   
    plt.show()
        
    
    # scaling
    plt.plot(x, y)
    ax = plt.gca()
    # values: log, logit, symlog
    ax.set_yscale("log")
    plt.grid(True)
    plt.title("Log Scaled Axis")
    plt.show()
    
    
    # annotations
    y = x**2
    min_x = 0
    min_y = min_x**2
    
    plt.plot(x, y, "b-", min_x, min_y, "ro")
    plt.axis([-10,10,-25,100])
    
    plt.text(0, 60, "Parabola\n$y = x^2$", fontsize=15, ha="center")
    plt.text(min_x, min_y+2, "Minima", ha="center")
    plt.text(min_x, min_y-6, "(%0.1f, %0.1f)"%(min_x, min_y), ha='center',color='gray')
    plt.title("Annotated Plot")
    plt.show()
    
    
    # global formatting params
    params = {'legend.fontsize': 'large',
              'figure.figsize': (10, 10),
             'axes.labelsize': 'large',
             'axes.titlesize':'large',
             'xtick.labelsize':'large',
             'ytick.labelsize':'large'}

    plt.rcParams.update(params)
    
    
    # saving
    #plt.savefig("sample_plot.png", transparent=True)