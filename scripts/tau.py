#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import itertools
import os
import sys

def noWindow(data,jfileName,calcIndex):
    # unpack the json file
    tau = np.array(data['relaxationTimes'])    # dimensions (iCalc, ik, ib)
    energies = np.array(data['energies'])      # dimensions (iCalc, ik, ib)
    nbands = energies.shape[2]
    mu = np.array(data['chemicalPotentials'])
    T = np.array(data['temperatures'])

    energies = energies[calcIndex].flatten()
    tau = tau[calcIndex].flatten()
    mu = mu[calcIndex]

    print("Calculation Temperature: ", T[calcIndex])

    return energies-mu, tau, linewidths

def withWindow(data,jfileName,calcIndex):
    # unpack the json file
    srcTau = np.array(data['relaxationTimes'])    # dimensions (iCalc, ik, ib)
    srcEnergies = np.array(data['energies'])      # dimensions (iCalc, ik, ib)
    srcLinewidths = np.array(data['linewidths'])      # dimensions (iCalc, ik, ib)
    tau = []
    energies = []
    linewidths = []
    for x in srcTau:
        tau.append(np.array(list(itertools.chain(*x))))
    for x in srcEnergies:
        energies.append(np.array(list(itertools.chain(*x))))
    for x in srcLinewidths:
        linewidths.append(np.array(list(itertools.chain(*x))))
    mu = np.array(data['chemicalPotentials'])
    T = np.array(data['temperatures'])
    
    print("Calculation Temperature: ", T[calcIndex])

    return energies[calcIndex]-mu[calcIndex], tau[calcIndex], linewidths[calcIndex]
        

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plot relaxation times that "
                                     "have been generated by Phoebe")
    parser.add_argument("INPUT",
                        help="Name of the JSON file with relaxation times")
    parser.add_argument("calcIndex",
                        help="Index of temperature/chemical potential to plot",
                        default=0)
    args = parser.parse_args()

    # we select one calculation
    try:
        calcIndex = int(args.calcIndex)
    except ValueError:
        raise ValueError("calcIndex should be an integer")

    # load in the json output
    jfileName = args.INPUT
    with open(jfileName) as jfile:
        data = json.load(jfile)

        # some relaxation times may be None (e.g. acoustic phonon modes at Gamma)
        # we replace that with 0, in order to be plotted
        try:
            data['relaxationTimes'] =[ [ [ 0. if x==None else x for x in y]
                                         for y in z ]
                                       for z in data['relaxationTimes'] ]
        except KeyError:
            raise KeyError("relaxation times not found."
                           "Are you using the correct input json file?")
        
        
    if len(np.array(data['relaxationTimes']).shape) == 3:
        energies, tau, linewidths = plotNoWindow(data,jfileName,calcIndex)
    else:
        energies, tau, linewidths = withWindow(data,jfileName,calcIndex)
        
    # Could also load in group velocities, or wavevectors

    for y, name in [[tau,'tau'],
                    [linewidths,'Gamma']]:    
        # plot the lifetimes
        plt.figure(figsize=(5,5))
        plt.scatter(energies, y,
                    marker='o', s=18, color='royalblue')
        
        # plot aesthetics
        plt.yscale('log')
        plt.xlabel(r'Energy [' + data['energyUnit'] +']',fontsize=12)
        if name == 'tau':
            units = ' [' + data['relaxationTimeUnit'] + ']'
        else:
            units = ' [' + data['linewidthsUnit'] + ']'
            
        plt.ylabel(r'$\{}_{{'.format(name) + data['particleType'] + '}$' +
                   units, fontsize=12)

        plt.xlim(0,None)

        # Find limits of the y axis
        zeroIndex = np.argwhere(y<=0.)
        y = np.delete(y, zeroIndex)
        print(y.shape)
        ymin = 10**np.floor(np.log10(np.min(y)))
        ymax = 10**np.ceil(np.log10(np.max(y)))
        plt.ylim(ymin, ymax)

        # plt.colorbar()
        
        plt.tight_layout()
        
        plotFileName = os.path.splitext(jfileName)[0] + ".{}.pdf".format(name.lower())
        plt.savefig(plotFileName)
