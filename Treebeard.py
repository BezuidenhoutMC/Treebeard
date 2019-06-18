#!/usr/bin/env python
# coding: utf-8

# Tiaan Bezuidenhout, 03/2019
# Code to sort through singlepulse candidates output by AstroAccelerate
# & identify those most likely to be astrophysical. 

# NB:
# Candidate files are assumed to consist of 4 columns: DM; SNR; Time; Boxcar Width

#-----------------------------------------------------------------------
# # Import

import numpy as np
import math
import matplotlib
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram,linkage,fcluster
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
import os
import gc
import argparse
import optparse
import sys

from matplotlib.cm import ScalarMappable
np.set_printoptions(threshold=np.nan)


#--------------------------------------------------------------------------
# Reads .dat files output from AstroAccelerate
def read_sps(file,dmrange):
	sps = np.genfromtxt(file,delimiter=',')
	try:
		sps = sps[sps[:,0]>=dmrange[0]]	
		sps = sps[sps[:,0]<=dmrange[1]]	
	except:
		print("NOTE: There's only one candidate in this file. Cannot do anything with it.")
	
	numcands=len(sps)
	print("Number of candidates between DM limits: "+str(numcands))
	time_col=2
	dm_col=0
	return sps,time_col,dm_col,numcands


#--------------------------------------------------------------------------
# Plot singlepulses DM vs Time

def plot_sps(sps,t_col,dm_col):
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    plt.scatter(sps[:,t_col],sps[:,dm_col])
#     ax.set_yscale("log")
    plt.draw

#---------------------------------------------------------------------------
# Cluster singlepulse candidates in time

def time_clust(sps,t,m,dist,chunks):
    if chunks > 1:
	for  c in range(1,chunks+1):
	   print("	Clustering chunk #"+str(c))
	   full_Z=[]
	   chunk_tstart = int(math.ceil(len(sps)/chunks)*(c-1))
	   chunk_tend = int(math.ceil(len(sps)/chunks)*c)
	   datachunk=sps[chunk_tstart:chunk_tend]
	   Z = linkage(datachunk[:,[t]],method=m)
	   full_Z.append(Z)
	   clusters = fcluster(Z, dist, criterion="distance")
	   if c==1:
	       full_clusters=clusters.tolist()
	   else:
	       clusters=(clusters+max(full_clusters)).tolist()
	       full_clusters =full_clusters+clusters
	full_clusters=np.asarray(full_clusters)
    if chunks == 1:
       full_Z = linkage(sps[:,[t]],method=m)
       full_clusters = fcluster(full_Z, dist, criterion="distance")
    return full_clusters,full_Z

#---------------------------------------------------------------------------
# Plot a dendrogram of clusters

def plot_dendro(Z):
    plt.figure(figsize=(30, 10))
    plt.title('SP Clustering Dendrogram')
    plt.xlabel('Single pulse clusters')
    plt.ylabel('distance')
    dendrogram(Z,truncate_mode='lastp',color_threshold=0)
    plt.show()

#-------------------------------------------------------------------------
# Cluster candidate time-clusters in DM
 
def dm_clust(sps,t_clust,dm,dist_thresh):   #(singlepulse array, time clusters, dm column number, dm clustering threshold)

    all_clust=np.zeros(len(t_clust))   # Each entry in all_clust[] records the final cluster number of the corresponding entry in sps[]
    all_clust[:]=np.nan

    n = 0 # Keeps track of position in all_clust[] while iterating
    for i in range(1,max(t_clust)+1): # For each time cluster
        clus3_idx = np.nonzero(t_clust==i)  # Indices in sps for cluster i
        clus3 = sps[clus3_idx]	# Values in sps for cluster i
        if len(clus3) < 2:  # Cannot further cluster time-clusters of size 1
            all_clust[clus3_idx] = 0 
            n+=1
        else:
            Z2 = linkage(clus3[:,[dm]],method='single')
            dubclus = fcluster(Z2, dist_thresh, criterion="distance") #DM clusters for time cluster i
            dubclus=np.array(dubclus)+n # Ensures all clusters have different number
            n=max(dubclus)

            all_clust[clus3_idx] = dubclus 
    all_clust=all_clust.astype(int)

    return all_clust

#--------------------------------------------------------------------------
# Marks clusters as RFI if they are very small or if the DM is very low
# Cluster 0 is RFI & noise

def flag_rfi(sps, all_clust,dm_thresh,clust_size_thresh,width_thresh):
    for i in range(min(all_clust),max(all_clust)+1):        #SMALL CLUSTERS

	dms=[]
	widths=[]
	for x in np.nonzero(all_clust==i)[0]:  #For each SP of cluster i
            dms.append(sps[x][0])       # width of each SP in cluster i
	    widths.append(sps[x][3])
        if len(np.nonzero(all_clust==i)[0])<clust_size_thresh:
            rfi_flag = np.nonzero(all_clust==i)[0]
            all_clust[rfi_flag] = 0
        
        else:
            #print(sps[np.nonzero(all_clust==i)][:][1])   
	    #LOW DM
            if max(dms) < dm_thresh:
#                 print("Flagged as RFI")
                rfi_flag = np.nonzero(all_clust==i)[0]
                all_clust[rfi_flag] = 0
	    #WIDE
	    elif max(widths) > width_thresh: 
		rfi_flag = np.nonzero(all_clust==i)[0]
                all_clust[rfi_flag] = 0
	    #print(sps[np.nonzero(all_clust==i)][0][3])
        
    return all_clust

#-----------------------------------------------------------------------------
# Plot clusters (DM vs Time) with different colours for different clusters


def plot_clust2(sps,cl,t,dm,annotate,yplotrange,outfile):  #(singlepulses, clusters, time column, DM column)
    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    plt.title('SP Clusters')
    plt.xlabel('Time')
    plt.ylabel('DM')
	
    if yplotrange!=[]:
        ax.set_ylim(yplotrange)

    for i in range(min(cl),max(cl)+1):
	q = np.nonzero(cl==i)[0]
	if i==0:
	    plt.scatter(sps[q][:,t],sps[q][:,dm],color="gray",marker='*',s=1)
	else:
	    points = plt.scatter(sps[q][:,t],sps[q][:,dm],s=15)
        if len(q) !=0 and annotate:
            ax.annotate(str(i),(sps[q][0,t],sps[q][0,dm]))




    plt.savefig("Clusters_"+outfile+".png")
    plt.draw()



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Rank clusters from 1 to 6, from bad to good.

def rank_groups(sps,groups,plot_sps,outfile):
    min_group = 2
    sigma_thresh = 5
    
    sigmas_max=np.zeros(max(groups)+1)
    sigmas_max_idx=np.zeros(max(groups)+1)
    opt_DM=np.zeros(max(groups)+1)
    opt_width=np.zeros(max(groups)+1)
    numsps=np.zeros(max(groups)+1)    
    ranks = np.zeros(max(groups)+1)
    ranks[:] = 1

    
    ranks_full_arr = np.zeros(len(sps))
    ranks_full_arr[:] = 1
#     colours = []
    
    rank_to_colour = {2:'darkgrey', 0:'k', 3:'c', 4:'royalblue', 5:'g', 6:'m',1:'r'}
    if plot_sps==True:
        print('	Saving ranking diagnostic plots...') 
#------------------
    for i in range(1,max(groups)+1):   #For each cluster numbered i     
        sigmas = []
        dms = []
	widths = []
        idx=[]
	times=[]
        
        numsps[i] = len(np.nonzero(groups==i)[0])
        
        for x in np.nonzero(groups==i)[0]:  #For each SP of cluster i
            idx.append(x)                  # Indices of sps in cluster i
            sigmas.append(sps[x][1])       # SNR of each SP in cluster i
            dms.append(sps[x][0])       # DM of each SP in cluster i
	    widths.append(sps[x][3])    # Width of each SP in cluster i
	    times.append(sps[x][2])     # Time stamp of each SP in cluster i
            
        try:
            sigmas_max[i]=max(sigmas)          # Maximum SNR in cluster i
            sigmas_max_idx[i] = idx[np.where(sigmas==sigmas_max[i])[0][:][0]] # First index corresponding to maximum SNR in cluster i
            opt_DM[i] = sps[int(sigmas_max_idx[i]),0]       #DM at max SNR in cluster i
	    opt_width[i] = sps[int(sigmas_max_idx[i]),3]    #Width at max SNR in cluster i
        except:                #If zero sps in cluster i
            sigmas_max[i]=0 # Empty clusters assigned SNR=0               
            sigmas_max_idx[i]=np.nan
            opt_DM[i] = np.nan
            opt_width[i] = np.nan

       	best_group = np.nonzero(widths==opt_width[i])[0]        # All SPs in cluster with optimal width 
	best_dms = [dms[x] for x in best_group] 			       # DMs of SPs with optimal width
	best_sigmas = [sigmas[x] for x in best_group]   
	best_widths = [widths[x] for x in best_group]
#-------------------------------------    
#         RANKING ALGORITHM   (From RRATtrap.py)
#-------------------------------------


        if numsps[i] < min_group and ranks[i] != 7:   #Rank group as 1 if less numerous than threshold
            ranks[i] = 1

        elif ranks[i] != 2 and ranks[i] != 7 :
            sigmas_sorted = [x for _,x in sorted(zip(dms,sigmas))]
            sigmas_arr = np.zeros(np.int(np.ceil(numsps[i]/5)*5))
            sigmas_arr[-np.int(numsps[i]):] = sigmas            # Array of cluster SNRs sorted by DM from low to high
            
            sigmas_arr.shape = (5,np.int(np.ceil(numsps[i]/5)))  # sigmas_arr reshaped to have 5 rows
            
	#---------------------------------------------------------------
            maxsigmas = np.max(sigmas_arr,axis=1)        # max SNR in each row
            avgsigmas = np.mean(sigmas_arr,axis=1)       # mean SNR in each row
            stdsigmas = np.std(sigmas_arr,axis=1)        # standard deviation of SNR in each row 
            maxstd = np.max(stdsigmas)                   # maximum standard deviation of all rows
            minstd = np.min(stdsigmas)
            maxsigma = np.max(maxsigmas)
            minsigma = np.min(maxsigmas)
            maxavgsigma = np.max(avgsigmas)
            minavgsigma = np.min(avgsigmas)
	#--------------------------------------------------------------
        # RANKING LOGIC

            if all(std < 0.1 for std in stdsigmas):
                ranks[i]=1                   # Mark cluster i as RFI (rank 1) if sigmas pretty much constant
            if maxsigmas[2] > maxsigmas[1]:
                if maxsigmas[2] > maxsigmas[3]:
                    ranks[i] = 3
                    if (maxsigmas[3] > maxsigmas[4]) and (maxsigmas[1] > maxsigmas[0]): 
                        ranks[i] = 4
                        if maxsigmas[2] > sigma_thresh:  
                            # We want the largest maxsigma to be at least 
                            # 1.15 times bigger than the smallest
                            ranks[i] = 5
                            if (avgsigmas[2] > avgsigmas[0]) and (avgsigmas[2] > avgsigmas[4]) and maxsigma>1.15*minsigma:
                                    ranks[i] = 6 
                else: #ie. maxsigmas[2] <= maxsigmas[3], allowing for asymmetry:
#                     print("Entered path maxsigmas[2]<=maxsigmas[3]")
                    if maxsigmas[1] > maxsigmas[0]:
                        ranks[i] = 3
                        if maxsigmas[3] > maxsigmas[4]:
                            ranks[i] = 4
                            if maxsigmas[3] > sigma_thresh:
                                ranks[i] = 5
                                if (avgsigmas[3] > avgsigmas[0]) and                                     (avgsigmas[3] > avgsigmas[4]) and                                     maxsigma>1.15*minsigma:
                                    ranks[i] = 6 

            else: #ie. maxsigma2 >= maxsigma3, allowing for asymmetry:
#                 print("Entered path maxsigma[2] <= maxsigma[1]")
                if (maxsigmas[1] > maxsigmas[0]) and (maxsigmas[2] > maxsigmas[3]):
                    ranks[i] = 3
                    if maxsigmas[3] > maxsigmas[4]:
#                         print("Entered path maxsigmas[3]>maxsigmas[4]")
                        ranks[i] = 4
                        if maxsigmas[1] > sigma_thresh:
#                             print("Entered path maxsigmas[1]>sigma_thresh")
                            ranks[i] = 5
                            if (avgsigmas[1] >= avgsigmas[0]) and                                 (avgsigmas[1] > avgsigmas[4]) and                                 maxsigma>1.15*minsigma:
                                ranks[i] = 6
            if any(stdsigma < 0.1 for stdsigma in stdsigmas) and (sigmas_max[i] < 5.5): # if max sigma of the group is less than 5.5 and the sigma distribution is mostly flat, then it is not likely to be astrophysical.
                ranks[i] = 1
            if ranks[i] == 0:
                pass

	#--------------------------------------------------------------
            
            if plot_sps==True: 
                #----------------------------------------------------#
		fig = plt.figure(figsize=(10, 8))              #PLOT SPs
                ax = plt.gca()
		sc = plt.scatter(dms,sigmas,c = widths)
		plt.axvline(x=opt_DM[i])
		plt.scatter(best_dms,best_sigmas,color='red')
		cbar = plt.colorbar(sc)
		cbar.set_label('Pulse Width')
		plt.text(min(dms),max(sigmas),"Cluster "+str(i)+"\nTime: %.3f" % min(times),verticalalignment='top',bbox=dict(facecolor=rank_to_colour[ranks[i]],alpha=0.5))
		#plt.text(max(dms),max(sigmas),"Time: "+str(min(times)),bbox=dict(facecolor=rank_to_colour[ranks[i]],alpha=0.5))
                plt.ylabel("SNR")
                plt.xlabel("DM")
		plt.savefig("cl"+str(i)+'_'+outfile+".png")
                #----------------------------------------------------# 

        ranks_full_arr[np.nonzero(groups==i)] = ranks[i]
          
    return ranks_full_arr #Array of rankings (1-6) for each SPS


#--------------------------------------------------------------------------------------------------------------------------------------------
# Plots SP DM vs Time with colours based on ranking 
def plotranks(file,sps,ranks_arr,filename,yplotrange,outfile):
    rank_to_colour = {2:'darkgrey', 0:'k', 3:'c', 4:'royalblue', 5:'g', 6:'m',1:'r'}
    rank_size={1:0.1,2:15,3:15,4:15,5:20,6:70}
    sizes=[]
    colours=[]
#     print("KEY: \n 0 - Black \n 1 - Red \n 2 - Gray \n 3 - Cyan \n 4 - Blue \n 5 - Green \n 6 - Magenta")
    for q in range(0,len(sps)):
            colours.append(rank_to_colour[ranks_arr[q]])
	    sizes.append(rank_size[ranks_arr[q]])
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    plt.xlabel("Time (s)")
    plt.ylabel("DM")
    plt.title(filename)
    if yplotrange!=[]:
    	ax.set_ylim(yplotrange)
    plt.scatter(sps[:,2], sps[:,0], c=colours,s=sizes, cmap='Set1')
    plt.savefig("Ranks_"+outfile+".png")
    #plt.draw()

#-------------------------------------------------------------------------------------------------------------------
# Searches through any number of output files from AstroAccelerate and assigns ranks to each SP found
def sps_search(inputfiles,dmrange,t_dist,dm_dist,chunks,plotclusters,flagrfi,rfidm,rfisize,rfiwidth,annotate,yplotrange,autochunk):
	n=1
#    print("KEY: \n 0 - Black \n 1 - Red (RFI) \n 2 - Gray \n 3 - Cyan \n 4 - Blue \n 5 - Green \n 6 - Magenta")
#     print("CPU usage: "+str(psutil.cpu_percent())+"%")
	for file in inputfiles:
		outfile=file
		print(n)
#         print("-----------------------------------------------------------")

		if os.stat(file).st_size !=0:
			print("Preparing file: "+file+"...")
			singlepulse,t_column,dm_column,numcands = read_sps(file,dmrange)
#             print(singlepulse)
#             plot_sps(singlepulse,t_column,dm_column)

			if len(singlepulse)>1:
				if autochunk==True:
					#print(numcands)
					chunks=int(math.ceil(numcands/20000.0))
					#print(chunks)
				print("Clustering in time...")
				time_clusters,z = time_clust(singlepulse,t_column,"ward",t_dist,chunks)
				print("Number of time clusters: "+str(max(time_clusters)))
			        #     plot_dendro(z)

				print("Clustering in DM...")
				dm_clusters = dm_clust(singlepulse,time_clusters,dm_column,dm_dist)
     				if flagrfi==True:
					dm_clusters = flag_rfi(singlepulse,dm_clusters,rfidm,rfisize,rfiwidth)
				plot_clust2(singlepulse,dm_clusters,t_column,dm_column,annotate,yplotrange,outfile)
				print("Total number of clusters: "+str(max(dm_clusters)))
				if len(dm_clusters) !=0:
					print("Ranking clusters...")
					ranks = rank_groups(singlepulse,dm_clusters,plotclusters,outfile)        
					plotranks(file,singlepulse,ranks,file,yplotrange,outfile)
					if max(ranks) > 3: 
						print("Found some clusters above rank 3!")

					else:
						print("No candidates above rank 3.")
	
		n+=1 
#         print("RAM usage: "+str(psutil.virtual_memory()[2])+"%")
		print("----------------------------------------------------------------")
	print("KEY: \n 0 - Black \n 1 - Red \n 2 - Gray \n 3 - Cyan \n 4 - Blue \n 5 - Green \n 6 - Magenta")

#-----------------------------------------------------------------------------------------------------------------------------



def main():
	parser = argparse.ArgumentParser(description='Run Treebeard singlepulse clustering & ranking.')

	parser.add_argument('-f', dest='filelist', nargs = '+', type = str, help="Any number of candidate files",required=True)
	parser.add_argument('--dmrange',dest='dmrange',type=float,nargs=2, help='Limits the DM range to search for pulses',default=[0,10000000])
	parser.add_argument('--tdist', dest='t_dist', type = float, help="Sets the granularity of time clustering", default=0.1)
	parser.add_argument('--dmdist', dest='dm_dist', type = float, help="Sets the granularity of DM clustering", default=50)
	parser.add_argument('--c',dest='chunks',type = int,help="Breaks the data up into a number of chunks before doing time clustering. Note: only do this with time-sequential data",default=1)
	parser.add_argument('--plotclusters',action="store_true",dest='plotclusters',help="Makes plots of each cluster's SNR vs DM",default=False)
	parser.add_argument('--norfi',action=
"store_false",dest='flagrfi',help='Don\'t remove RFI based on low DM & small cluster size.',default=True)
	parser.add_argument('--RFIdm',dest='rfidmthresh',type=float,help="Marks all candidates with DM lower than this as RFI",default=10)
	parser.add_argument('--RFIsize',dest='rfisizethresh',type=int,help="Marks clusters with fewer candidates than this as RF",default=4)
	parser.add_argument('--RFIwidth',dest='rfiwidththresh',type=float,help="Marks cluster wider than this (in ms) as RFI",default=100000000)
	parser.add_argument('--annotate',dest='annotate',action='store_true',help="Adds cluster numbers to the cluster plot",default=False)
	parser.add_argument('--yrange',dest='yplotrange',nargs=2,type=float,help="DM range for plots", default = [])
	parser.add_argument('--autochunk',dest='autochunk',action='store_true',help='Automatically chunks the data to process 20,000 candidates at a time',default=False)
	parser.add_argument('--o',dest='outfile',type=str,help='Name of output file. Default: input file + .png',default='')


	options= parser.parse_args()

	sps_search(options.filelist,options.dmrange,options.t_dist,options.dm_dist,options.chunks,options.plotclusters,options.flagrfi,options.rfidmthresh,options.rfisizethresh,options.rfiwidththresh,options.annotate,options.yplotrange,options.autochunk)


main()



