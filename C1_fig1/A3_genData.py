import numpy as np
import math 
from datetime import datetime
seedVal = 12345

def genNewData(seedMe, a, b, decayType, filename):
    numFeats = a
    numSamps = b
    # perturbation = np.random.rand(numSamps, numFeats)*0.1
    np.random.seed(seedMe)
    A1 = np.random.rand(numSamps, numFeats) 
    Q1, R1 = np.linalg.qr(A1)
    # Q1 = Q1 + perturbation
    A2 = np.random.rand(numFeats, numFeats) #Modify
    Q2, R2 = np.linalg.qr(A2)
    S = list(np.random.rand(numFeats)) #Modify
    S = S[::-1]
    for j in range(numFeats): #Modify
        # S[j] = 2**(poly(j))
        # poly(j) :
        # -j^2
        # -j*1000
        # (j-1000)^2 - 1000^2
        
        # if decayType == 'bot':
        #     S[j] = 2**(-(((j+1)/numFeats)**(1/2))) #SCALING RUN BOT
        # elif decayType == 'top':
        #     S[j] = 2**(-(((j+1)/numFeats)**2)) #SCALING RUN MID
        # elif decayType == 'mid':
        #     S[j] = 2**(-(j+1)/numFeats)
        # else:
        #     print("DECAY TYPE NOT RECOGNIZED")

        if decayType == 'top':
            S[j] = 0.5**(((j/150)**2)) #SCALING RUN BOT
        elif decayType == 'mid':
            S[j] = 0.5**(j/22.5) #SCALING RUN MID
        elif decayType == 'bot':
            S[j] = 0.5**(-((1000 - j)**2 - 1000**2)/150**2)
        else:
            print("DECAY TYPE NOT RECOGNIZED")
        
        #This is expDecaying underscore new 
        # if decayType == 'top':
        #     S[j] = 0.5**(((j/500)**5)) #SCALING RUN BOT
        # elif decayType == 'mid':
        #     S[j] = 0.5**(j/31) #SCALING RUN MID
        # elif decayType == 'bot':
        #     S[j] = 0.5**(-((1000 - j)**5 - 1000**5)/500**5)
        # else:
            # print("DECAY TYPE NOT RECOGNIZED")

        #SMALL CASE
        # if decayType == 'top':
        #     S[j] = 0.5**(((j/15)**2)) #SCALING RUN BOT
        # elif decayType == 'mid':
        #     S[j] = 0.5**(j/2.25) #SCALING RUN MID
        # elif decayType == 'bot':
        #     S[j] = 0.5**(-((100 - j)**2 - 100**2)/15**2)
        # else:
        #     print("DECAY TYPE NOT RECOGNIZED")
        #This is for testing new data
        # if decayType == 'top':
        #     S[j] = 0.5**(((j/50)**5)) #SCALING RUN BOT
        # elif decayType == 'mid':
        #     S[j] = 0.5**(j/3.1) #SCALING RUN MID
        # elif decayType == 'bot':
        #     S[j] = 0.5**(-((100 - j)**5 - 100**5)/50**5)
        # else:
        #     print("DECAY TYPE NOT RECOGNIZED")



        # if decayType == 'top':
        #     S[j] = 2**(13) - 2**(13*(j+1)/len(S)) #SCALING RUN TOP
        # elif decayType == 'bot':
        #     S[j] = 2**(13*(j+1)/len(S)) #SCALING RUN MID
        # elif decayType == 'mid':
        #     S[j] = 2**(13)*(j+1)/len(S) #SCALING RUN BOT
        # else:
        #     print("DECAY TYPE NOT RECOGNIZED")

        # S[j] = (2**(-16*(j+1)/len(S)))*S[j] #SCALING RUN
    # print(S)
    S = np.diag(S)
    fin = (Q1 @ S @ Q2).T
    np.save(filename, fin)

# matMe = genNewData(seedVal, 12000, 120000)
# matMe = genNewData(seedVal, 1000, 15000)

genNewData(seedVal, 1000, 15000, decayType='top',
filename = '/sdf/home/w/winnicki/papertests_20240717/expDecayingSingularValues_top.npy')
genNewData(seedVal, 1000, 15000, decayType='mid',
filename = '/sdf/home/w/winnicki/papertests_20240717/expDecayingSingularValues_mid.npy')
genNewData(seedVal, 1000, 15000, decayType='bot',
filename = '/sdf/home/w/winnicki/papertests_20240717/expDecayingSingularValues_bot.npy')

# genNewData(seedVal, 100, 1500, decayType='top',
# filename = '/sdf/home/w/winnicki/papertests_20240717/expDecayingSingularValues_topsmall.npy')
# genNewData(seedVal, 100, 1500, decayType='mid',
# filename = '/sdf/home/w/winnicki/papertests_20240717/expDecayingSingularValues_midsmall.npy')
# genNewData(seedVal, 100, 1500, decayType='bot',
# filename = '/sdf/home/w/winnicki/papertests_20240717/expDecayingSingularValues_botsmall.npy')

print("Finished!")
