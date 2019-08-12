# Treebeard
Tiaan Bezuidenhout, 2019

Code to classify candidate single pulses output from an AstroAccelerate single-pulse search.
Performs hierarchical clustering in first the time and then DM axes, and then assigns each cluster
a rank from 0 to 6 based on the shape of the pulse in SNR vs DM. Requires candidate list with columns:
DM, SNR, Time, Boxcar Width  in that order. 

Usage: python Treebeard.py [options] -f [files]

Run python Treebeard.py -h for a list of options.
