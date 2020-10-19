#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plot relaxation times that "
                                     "have been generated by Phoebe")
    parser.add_argument("INPUT",
                        help="Name of the JSON file with relaxation times")
    args = parser.parse_args()
    
    # load in the json output
    jfileName = args.INPUT
    with open(jfileName) as jfile:
        data = json.load(jfile)

    # unpack the json file
    try:
        kappa = np.array(data['thermalConductivity'])
        # size: (temperature, dim1, dim2)
    except KeyError:
        raise KeyError("Thermal conductivity not found."
                       "Are you using the correct input json file?")

    T = np.array(data['temperatures'])

    # plot the thermal conductivity (here we just plot xx component)
    plt.plot(T, kappa[:,0,0], lw=2, mew=1.5, ms=8, marker='o', color='royalblue')
    
    plt.xlabel('Temperature [' + data['temperatureUnit'] + ']',fontsize=12)
    plt.ylabel(r'$\kappa_\mathrm{' + data['particleType'] +
               '}$ [' + data['thermalConductivityUnit'] +']',fontsize=12)
    plt.ylim(None, np.max(kappa) + np.max(kappa)*0.1)
    plt.xlim(None, np.max(T) + np.max(T)*0.1)
    
    plt.tight_layout()

    plotFileName = "./" + jfileName.rstrip(".json")+".pdf"
    plt.savefig(plotFileName)
    plt.show(block=False)
    
