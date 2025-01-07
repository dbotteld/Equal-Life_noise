# try a more clever scaler with mean and stdev on each group of features
# reorganise the data such that all features of the same type are grouped
# refer to the document ML_input_indicator_range.xlsx on sharepoint for range definitions
import numpy as np

class GroupScaler:
    def __init__(self):
        self.inputscalers = []
    
    def fit(self, samples): 
        Nsamples = samples.shape[0]
        samples_reordered = []
        #building densities
        samples_reordered.append(np.reshape(samples[:,0:36],(36*Nsamples,)))
        #distances within 200m
        samples_reordered.append(np.reshape(samples[:,36:432],(((432-36)*Nsamples,))))
        #building density
        samples_reordered.append(samples[:,432])
        #road length
        samples_reordered.append(np.reshape(samples[:,433:437],(((437-433)*Nsamples,))))
        #building density
        samples_reordered.append(samples[:,437])
        #road length
        samples_reordered.append(np.reshape(samples[:,438:442],(((442-438)*Nsamples,))))
        #building density
        samples_reordered.append(samples[:,442])
        #road length
        samples_reordered.append(np.reshape(samples[:,443:447],(((447-443)*Nsamples,))))
        #building density
        samples_reordered.append(samples[:,447])
        #road length
        samples_reordered.append(np.reshape(samples[:,448:452],(((452-448)*Nsamples,))))

        # now create separate scalers for each group
        for input_d in range(0,len(samples_reordered)): 
            self.inputscalers.append(StandardScaler(with_mean=False))
            self.inputscalers[input_d].fit(samples_reordered[input_d].reshape(-1,1))
    
    def transform(self, samples):
        Nsamples = samples.shape[0]
        samples_reordered = []
        #building densities
        samples_reordered.append(np.reshape(samples[:,0:36],(36*Nsamples,)))
        #distances within 200m
        samples_reordered.append(np.reshape(samples[:,36:432],(((432-36)*Nsamples,))))
        #building density
        samples_reordered.append(samples[:,432])
        #road length
        samples_reordered.append(np.reshape(samples[:,433:437],(((437-433)*Nsamples,))))
        #building density
        samples_reordered.append(samples[:,437])
        #road length
        samples_reordered.append(np.reshape(samples[:,438:442],(((442-438)*Nsamples,))))
        #building density
        samples_reordered.append(samples[:,442])
        #road length
        samples_reordered.append(np.reshape(samples[:,443:447],(((447-443)*Nsamples,))))
        #building density
        samples_reordered.append(samples[:,447])
        #road length
        samples_reordered.append(np.reshape(samples[:,448:452],(((452-448)*Nsamples,))))

        # now apply separate scalers for each group
        scaled_reordered = []
        for input_d in range(0,len(samples_reordered)): 
            scaled_reordered.append(self.inputscalers[input_d].transform(samples_reordered[input_d].reshape(-1,1)))
        
        # restore original sample
        scaled = np.zeros((Nsamples,452))
        #building densities
        scaled[:,0:36] = np.reshape(scaled_reordered[0],(Nsamples,36))
        #distances within 200m
        scaled[:,36:432] = np.reshape(scaled_reordered[1],(Nsamples,(432-36)))
        #building density
        scaled[:,432] = scaled_reordered[2].reshape(-1,)
        #road length
        scaled[:,433:437] = np.reshape(scaled_reordered[3],(Nsamples,(437-433)))
        #building density
        scaled[:,437] = scaled_reordered[4].reshape(-1,)
        #road length
        scaled[:,438:442] = np.reshape(scaled_reordered[5],(Nsamples,(442-438)))
        #building density
        scaled[:,442] = scaled_reordered[6].reshape(-1,)
        #road length
        scaled[:,443:447] = np.reshape(scaled_reordered[7],(Nsamples,(447-443)))
        #building density
        scaled[:,447] = scaled_reordered[8].reshape(-1,)
        #road length
        scaled[:,448:452] = np.reshape(scaled_reordered[9],(Nsamples,(452-448)))
        
        return scaled
    
    def inverse_transform(self, samples):
        Nsamples = samples.shape[0]
        samples_reordered = []
        #building densities
        samples_reordered.append(np.reshape(samples[:,0:36],(36*Nsamples,)))
        #distances within 200m
        samples_reordered.append(np.reshape(samples[:,36:432],(((432-36)*Nsamples,))))
        #building density
        samples_reordered.append(samples[:,432])
        #road length
        samples_reordered.append(np.reshape(samples[:,433:437],(((437-433)*Nsamples,))))
        #building density
        samples_reordered.append(samples[:,437])
        #road length
        samples_reordered.append(np.reshape(samples[:,438:442],(((442-438)*Nsamples,))))
        #building density
        samples_reordered.append(samples[:,442])
        #road length
        samples_reordered.append(np.reshape(samples[:,443:447],(((447-443)*Nsamples,))))
        #building density
        samples_reordered.append(samples[:,447])
        #road length
        samples_reordered.append(np.reshape(samples[:,448:452],(((452-448)*Nsamples,))))

        # now apply separate scalers for each group
        scaled_reordered = []
        for input_d in range(0,len(samples_reordered)): 
            scaled_reordered.append(self.inputscalers[input_d].inverse_transform(samples_reordered[input_d].reshape(-1,1)))
        
        # restore original sample
        scaled = np.zeros((Nsamples,452))
        #building densities
        scaled[:,0:36] = np.reshape(scaled_reordered[0],(Nsamples,36))
        #distances within 200m
        scaled[:,36:432] = np.reshape(scaled_reordered[1],(Nsamples,(432-36)))
        #building density
        scaled[:,432] = scaled_reordered[2].reshape(-1,)
        #road length
        scaled[:,433:437] = np.reshape(scaled_reordered[3],(Nsamples,(437-433)))
        #building density
        scaled[:,437] = scaled_reordered[4].reshape(-1,)
        #road length
        scaled[:,438:442] = np.reshape(scaled_reordered[5],(Nsamples,(442-438)))
        #building density
        scaled[:,442] = scaled_reordered[6].reshape(-1,)
        #road length
        scaled[:,443:447] = np.reshape(scaled_reordered[7],(Nsamples,(447-443)))
        #building density
        scaled[:,447] = scaled_reordered[8].reshape(-1,)
        #road length
        scaled[:,448:452] = np.reshape(scaled_reordered[9],(Nsamples,(452-448)))
        
        return scaled
