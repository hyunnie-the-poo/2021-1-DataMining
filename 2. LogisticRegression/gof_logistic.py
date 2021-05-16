import numpy as np
import pandas as pd
import scipy.stats as stats

def HosmerLemeshow(model,Y): 
    pihat=model.predict() 
    pihatcat=pd.cut(pihat, np.percentile(pihat,[0,25,50,75,100]), labels=False, include_lowest=True)

    meanprobs =[0]*4 
    expevents =[0]*4
    obsevents =[0]*4 
    meanprobs2=[0]*4 
    expevents2=[0]*4
    obsevents2=[0]*4 

    for i in range(4):
        meanprobs[i]=np.mean(pihat[pihatcat==i])
        expevents[i]=np.sum(pihatcat==i)*np.array(meanprobs[i])
        obsevents[i]=np.sum(Y[pihatcat==i])
        meanprobs2[i]=np.mean(1-pihat[pihatcat==i])
        expevents2[i]=np.sum(pihatcat==i)*np.array(meanprobs2[i])
        obsevents2[i]=np.sum(1-Y[pihatcat==i]) 

    data1={'meanprobs':meanprobs,'meanprobs2':meanprobs2}
    data2={'expevents':expevents,'expevents2':expevents2}
    data3={'obsevents':obsevents,'obsevents2':obsevents2}
    m=pd.DataFrame(data1)
    e=pd.DataFrame(data2)
    o=pd.DataFrame(data3)

    tt=sum(sum((np.array(o)-np.array(e))**2/np.array(e))) 
    pvalue=1-stats.chi2.cdf(tt,2)

    return pd.DataFrame([[stats.chi2.cdf(tt,2).round(2), pvalue.round(2)]],columns = ["Chi2", "p - value"])