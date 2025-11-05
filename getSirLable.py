import Utils
import pandas as pd
import numpy as np

if __name__ == "__main__":

    # load real-world networks
    PowerGrid = Utils.load_graph('./Networks/real/powergrid.txt')
    GrQC = Utils.load_graph('./Networks/real/CA-GrQc.txt')
    Facebook = Utils.load_graph('./Networks/real/facebook_combined.txt')
    Hep = Utils.load_graph('./Networks/real/CA-HepTh.txt')
    LastFM = Utils.load_graph('./Networks/real/LastFM.txt')
    Vidal = Utils.load_graph('./Networks/real/vidal.txt')
    Politician = Utils.load_graph('./Networks/real/Politician.txt')
    NetScience = Utils.load_graph('./Networks/real/NetScience.txt')
    Faa = Utils.load_graph('./Networks/real/faa.txt')

    a_list = np.arange(1.0,2.0,0.1)

    Facebook_SIR = Utils.SIR_betas(Facebook,a_list,'./SIR results/Facebook/Facebook_')
    GrQC_SIR = Utils.SIR_betas(GrQC,a_list,'./SIR results/GrQ/GrQ_')
    Hep_SIR = Utils.SIR_betas(Hep,a_list,'./SIR results/Hep/Hep_')
    LastFM_SIR = Utils.SIR_betas(LastFM,a_list,'./SIR results/LastFM/LastFM_')
    Vidal_SIR = Utils.SIR_betas(Vidal,a_list,'./SIR results/vidal/vidal_')
    PowerGrid_SIR = Utils.SIR_betas(PowerGrid,a_list,'./SIR results/powergrid/powergrid_')
    NetScience_SIR = Utils.SIR_betas(NetScience,a_list,'./SIR results/NetScience/NetScience_')
    Politician_SIR = Utils.SIR_betas(Politician,a_list,'./SIR results/Politician/Politician_')
    Faa_SIR = Utils.SIR_betas(Faa,a_list,'./SIR results/Faa/Faa_')
