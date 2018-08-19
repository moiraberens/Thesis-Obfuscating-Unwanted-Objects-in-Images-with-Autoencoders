import glob
import pandas as pd
import h5py
import os
import matplotlib.pyplot as plt
import numpy as np


if __name__ is "__main__":
    for type_ in [2]:
        for weight in [0.4]:
            private_losses = []
            public_losses = []
            qr_losses = []
            copy_losses = []
            for file in glob.glob('final_results/type{}_GRL_propAdd_semi_blind/weight_{}_*/*.h5'.format(type_, weight)):
                with h5py.File(file, 'r') as data:
                    val_qr_loss = data['val_qr_loss'][:]
                    val_whole_loss = data['val_private_loss'][:]
                    val_copy_loss = (val_whole_loss-(300/3072)*val_qr_loss)/(2772/3072)
                    #index = np.where((val_qr_loss+val_copy_loss)==np.min((val_qr_loss+val_copy_loss)))[0][0]
                    index = np.where((val_whole_loss)==np.min((val_whole_loss)))[0][0]
                    print(index)
                    private_losses.append(data['val_private_loss'][index])
                    public_losses.append(data['val_public_loss'][index])
                    qr_losses.append(data['val_qr_loss'][index])
                    copy_losses.append(val_copy_loss[index])
            print('MEAN: type: {0}, weight: {1}, private_loss: {2:.4f}~{3:.4f}, public_loss: {4:.4f}~{5:.4f}, qr_loss: {6:.4f}~{7:.4f}, copy_loss: {8:.4f}~{9:.4f}'.format(type_, weight, np.mean(private_losses), np.std(private_losses), np.mean(public_losses), np.std(private_losses), np.mean(qr_losses), np.std(qr_losses), np.mean(copy_losses), np.std(copy_losses)))
            #print('MEAN: type: {0}, weight: {1}, private_loss: {2:.4f}, public_loss: {3:.4f}, qr_loss: {4:.4f}'.format(type_, weight, np.mean(private_losses), np.mean(public_losses), np.mean(qr_losses)))
            #print('MIN:  type: {0}, weight: {1}, private_loss: {2:.4f}, public_loss: {3:.4f}, qr_loss: {4:.4f}'.format(type_, weight, np.min(private_losses), np.min(public_losses), np.min(qr_losses)))
        
# =============================================================================
#     print('')
#     type_ = 3
#     weight = 0.5
#     private_losses = []
#     public_losses = []
#     qr_losses = []
#     copy_losses = []
#     with h5py.File('final_results/type{}_GRL_propAdd/weight_{}_number_0/metrics_epoch_99.h5'.format(type_, weight), 'r') as data:
#         val_qr_loss = data['val_qr_loss'][:]
#         val_whole_loss = data['val_private_loss'][:]
#         val_copy_loss = (val_whole_loss-(300/3072)*val_qr_loss)/(2772/3072)
#         #index = np.where((val_qr_loss+val_copy_loss)==np.min((val_qr_loss+val_copy_loss)))[0][0]
#         index = np.where((val_whole_loss)==np.min((val_whole_loss)))[0][0]
#         private_losses.append(data['val_private_loss'][index])
#         public_losses.append(data['val_public_loss'][index])
#         qr_losses.append(data['val_qr_loss'][index])
#         copy_losses.append(val_copy_loss[index])
#     print('MEAN: type: {0}, weight: {1}, private_loss: {2:.4f}~{3:.4f}, public_loss: {4:.4f}~{5:.4f}, qr_loss: {6:.4f}~{7:.4f}, copy_loss: {8:.4f}~{9:.4f}'.format(type_, weight, np.mean(private_losses), np.std(private_losses), np.mean(public_losses), np.std(private_losses), np.mean(qr_losses), np.std(qr_losses), np.mean(copy_losses), np.std(copy_losses)))
#     #print('MEAN: type: {0}, weight: {1}, private_loss: {2:.4f}, public_loss: {3:.4f}, qr_loss: {4:.4f}'.format(type_, weight, np.mean(private_losses), np.mean(public_losses), np.mean(qr_losses)))
#     #print('MIN:  type: {0}, weight: {1}, private_loss: {2:.4f}, public_loss: {3:.4f}, qr_loss: {4:.4f}'.format(type_, weight, np.min(private_losses), np.min(public_losses), np.min(qr_losses)))
#             
# =============================================================================
    
# =============================================================================
#         file = file.replace('conv_deconv', 'autoencoder')
#         model, _, depth, _, initial_filters, _, extra_block, _, batch_norm, _, same_batch= file.split('_')
#         model = model.split('\\')[1]
#         #bla = file.split('_')
#         df = df.append({'model': model, 'depth': depth, 'initial_filters': initial_filters, 'extra_block': extra_block, 'batch_norm': batch_norm, 'same_batch':same_batch, 'Lcopy - wanted': public_loss, 'Lwhole - unwanted':private_loss, 'Lobject - unwanted':qr_loss}, ignore_index=True)
#         df[['initial_filters','depth']] = df[['initial_filters','depth']].apply(pd.to_numeric)
# =============================================================================
        
        
# =============================================================================
#     #df.to_csv('data_experiment.csv', encoding='utf-8', index=False)
#         
#     df['Lcopy - unwanted'] = (df['Lwhole - unwanted'] - df['Lobject - unwanted']*0.1)/0.9 
#        
#     df['bottleNeck'] = (df['initial_filters'] * pow(2,df['depth']-1)) * pow(32/pow(2,df['depth']),2)
#         
#     fig = plt.figure(1) 
#     fig.clf()
#     fig, ax = plt.subplots(nrows=1, ncols=1, num=1)
#     ax = plt.scatter(x=df['Lcopy - unwanted'], y=df['Lcopy - wanted'], c='b')
#     ax = plt.plot(np.unique(df['Lcopy - unwanted']), np.poly1d(np.polyfit(df['Lcopy - unwanted'], df['Lcopy - wanted'], 1))(np.unique(df['Lcopy - unwanted'])), c='k')
#     ax = plt.xlabel('Lcopy - unwanted')
#     ax = plt.ylabel('Lcopy - wanted')
#     ax = plt.ylim((0, 0.1))
#     ax = plt.xlim((0, 0.1))
#     plt.show()
#     plt.close
#     
#     dfskip = df.loc[df['depth'] == 3]
#     dfskip = dfskip.loc[df['initial_filters'] == 32]
#     dfskip = dfskip.loc[df['batch_norm'] == 'False']
#     dfskip = dfskip.loc[df['extra_block'] == 'skip']
#     dfskip = dfskip.loc[df['same_batch'] == 'True']
#     
#     dfbatch = df.loc[df['depth'] == 3]
#     dfbatch = dfbatch.loc[df['initial_filters'] == 32]
#     dfbatch = dfbatch.loc[df['model'] == 'unet']
#     dfbatch = dfbatch.loc[df['extra_block'] == 'skip']
#     dfbatch = dfbatch.loc[df['same_batch'] == 'True']
#     
#     dfextra = df.loc[df['depth'] == 3]
#     dfextra = dfextra.loc[df['initial_filters'] == 32]
#     dfextra = dfextra.loc[df['model'] == 'unet']
#     #dfbatch = dfbatch.loc[df['extra_block'] == 'skip']
#     dfextra = dfextra.loc[df['same_batch'] == 'True']
#     dfextra = dfextra.loc[df['batch_norm'] == 'False']
#     
#     dfdepth = df.loc[df['initial_filters'] == 32]
#     dfdepth = dfdepth.loc[df['model'] == 'unet']
#     dfdepth = dfdepth.loc[df['extra_block'] == 'skip']
#     dfdepth = dfdepth.loc[df['same_batch'] == 'True']
#     dfdepth = dfdepth.loc[df['batch_norm'] == 'False']
#     dfdepth = dfdepth.sort_values(by=['depth'])
#     
#     fig = plt.figure(3)
#     fig.clf()
#     fig, ax2 = plt.subplots(nrows=1, ncols=1, num=3)
#     ax2 = plt.plot(dfdepth['depth'].values, dfdepth['Lcopy - wanted'].values)
#     ax2 = plt.plot(dfdepth['depth'].values, dfdepth['Lwhole - unwanted'].values)
#     ax2 = plt.plot(dfdepth['depth'].values, dfdepth['Lcopy - unwanted'].values)
#     ax2 = plt.plot(dfdepth['depth'].values, dfdepth['Lobject - unwanted'].values)
#     ax2 = plt.xlabel('Depth')
#     ax2 = plt.ylabel('Loss')
#     plt.xticks([2,3])
#     ax2 = plt.legend(['Lcopy/Lwhole - wanted', 'Lwhole - unwanted', 'Lcopy - unwanted', 'Lobject - unwanted'])
#     plt.show()
#     
#     
#     
#     
#     
#     dffilter = df.loc[df['depth'] == 3]
#     dffilter = dffilter.loc[df['model'] == 'unet']
#     dffilter = dffilter.loc[df['extra_block'] == 'skip']
#     dffilter = dffilter.loc[df['same_batch'] == 'True']
#     dffilter = dffilter.loc[df['batch_norm'] == 'False']
#     dffilter = dffilter.sort_values(by=['initial_filters'])
#     
#     fig = plt.figure(2)
#     fig.clf()
#     fig, ax2 = plt.subplots(nrows=1, ncols=1, num=2)
#     ax2 = plt.plot(dffilter['initial_filters'].values, dffilter['Lcopy - wanted'].values)
#     ax2 = plt.plot(dffilter['initial_filters'].values, dffilter['Lwhole - unwanted'].values)
#     ax2 = plt.plot(dffilter['initial_filters'].values, dffilter['Lcopy - unwanted'].values)
#     ax2 = plt.plot(dffilter['initial_filters'].values, dffilter['Lobject - unwanted'].values)
#     ax2 = plt.xlabel('Number of initial filters')
#     ax2 = plt.ylabel('Loss')
#     plt.xticks([8,16,32])
#     ax2 = plt.legend(['Lcopy/Lwhole - wanted', 'Lwhole - unwanted', 'Lcopy - unwanted', 'Lobject - unwanted'])
#     plt.show()
# =============================================================================
    
# =============================================================================
#     fig = plt.figure(2)
#     fig.clf()
#     fig, ax2 = plt.subplots(nrows=1, ncols=1, num=2)
#     ax2 = plt.scatter(x=df['Lcopy - unwanted'], y=df['Lobject - unwanted'], c='b')
#     ax2 = plt.plot(np.unique(df['Lcopy - unwanted']), np.poly1d(np.polyfit(df['Lcopy - unwanted'], df['Lobject - unwanted'], 1))(np.unique(df['Lcopy - unwanted'])), c='k')
#     ax2 = plt.xlabel('Lcopy - unwanted')
#     ax2 = plt.ylabel('Lobject - unwanted')
#     plt.show()
#     
#     partDF = df['Lobject - unwanted': 'bottleNeck']
#     
#     
#     depth1 = df[['Lobject - unwanted', 'bottleNeck']].loc[df['depth'] == 1]
#     bla = depth1.boxplot(by='bottleNeck').
#     
#     depth2 = df[['Lobject - unwanted', 'bottleNeck']].loc[df['depth'] == 2]
#     depth3 = df[['Lobject - unwanted', 'bottleNeck']].loc[df['depth'] == 3]
#     
#     partDF = df[['Lobject - unwanted', 'bottleNeck']]
#     partDF.boxplot(by='bottleNeck')
#     
#     
#     fig = plt.figure(3)
#     fig.clf()
#     fig, ax3 = plt.subplots(nrows=1, ncols=1, num=3)
#     ax2 = plt.boxplot(data_test)
#     #ax2 = plt.boxplot(df['Lobject - unwanted'].loc[df['bottleNeck'] == 512])
#     #ax2 = plt.boxplot(df['Lobject - unwanted'].loc[df['bottleNeck'] == 1024])
#     #ax2 = plt.boxplot(df['Lobject - unwanted'].loc[df['bottleNeck'] == 2048])
#     #ax2 = plt.boxplot(df['Lobject - unwanted'].loc[df['bottleNeck'] == 4096])
#     #ax2 = plt.boxplot(df['Lobject - unwanted'].loc[df['bottleNeck'] == 8192])
#     #ax3 = plt.scatter(df['bottleNeck'], df['Lobject - unwanted'])
#     #ax3 = plt.plot(x=df['bottleNeck'], y=df['Lcopy - unwanted'])
#     #ax2 = plt.plot(np.unique(df['Lcopy - unwanted']), np.poly1d(np.polyfit(df['Lcopy - unwanted'], df['Lobject - unwanted'], 1))(np.unique(df['Lcopy - unwanted'])), c='k')
#     #ax3 = plt.xlabel('Lobject - unwanted')
#     #ax3 = plt.ylabel('Loss')
#     plt.show()
# =============================================================================