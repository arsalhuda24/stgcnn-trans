import os
import math
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import argparse
import glob
import torch.distributions.multivariate_normal as torchdist
from utils import * 
from metrics import * 
from ST_transformer.model import social_stgcnn
import copy

def test(KSTEPS=20):
    global loader_test,model
    model.eval()
    ade_bigls = []
    fde_bigls = []
    raw_data_dict = {}
    step =0

    for batch in loader_test: 
        step+=1
        #Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
         loss_mask,V_obs,A_obs,V_tr,A_tr, frame_obs, frame_pred = batch
        # print(frame_obs)
        print("ARSALL")
        #print(frame_pred)
        # print("obs_traj", obs_traj)
        # print(frame_obs.shape)
        # print(obs_traj.shape)
        # print(V_obs.shape)

        # print(obs_traj.shape)
        #print(obs_traj)

        # print(pred_traj_gt.shape)
        #dec_inp = start_of_seq
        num_of_objs = obs_traj_rel.shape[1]
        # print("V_tr",V_tr.shape)
        #Forward
        #V_obs = batch,seq,node,feat
        #V_obs_tmp = batch,feat,seq,node
        start_of_seq = torch.Tensor([0, 0]).unsqueeze(0).unsqueeze(1).repeat(1, 12, V_tr.shape[2],1).cuda()
        # print("V_tr",V_tr.shape)
        # print("start_Seq",start_of_seq.shape)
        #a=qw
        V_obs_tmp =V_obs.permute(0,3,1,2)
        # print(V_obs_tmp.shape)
        # print(A_obs.squeeze().shape)

        # print(V_obs_tmp)
        # print(A_obs)
        V_pred,_ = model(V_obs_tmp,A_obs.squeeze(),V_tr)

        # print("V_pred",V_pred.shape)
        # torch.Size([1, 5, 12, 2])
        # torch.Size([12, 2, 5])

        V_pred = V_pred.unsqueeze(0)  # arsal

        V_pred = V_pred.permute(0, 2, 1, 3)  # arsal
        # V_pred = V_pred.permute(0,2,3,1)

       # V_pred = V_pred.permute(0,2,3,1)

        # torch.Size([1, 12, 2, 5])>>seq,node,feat
        # V_pred= torch.rand_like(V_tr).cuda()


        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze().cuda()
        num_of_objs = obs_traj_rel.shape[1]
        # print("V_pred1",V_pred.shape)

        V_pred,V_tr =  V_pred[:,:num_of_objs,:],V_tr[:,:num_of_objs,:]
        """ARSALLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL"""
        # print(V_pred.shape)
        #For now I have my bi-variate parameters
        #normx =  V_pred[:,:,0:1]
        #normy =  V_pred[:,:,1:2]
        sx = torch.exp(V_pred[:,:,2]) #sx
        sy = torch.exp(V_pred[:,:,3]) #sy
        corr = torch.tanh(V_pred[:,:,4]) #corr
        
        cov = torch.zeros(V_pred.shape[0],V_pred.shape[1],2,2).cuda()
        cov[:,:,0,0]= sx*sx
        cov[:,:,0,1]= corr*sx*sy
        cov[:,:,1,0]= corr*sx*sy
        cov[:,:,1,1]= sy*sy
        mean = V_pred[:,:,0:2]
        
        mvnormal = torchdist.MultivariateNormal(mean,cov)


        ### Rel to abs 
        ##obs_traj.shape = torch.Size([1, 6, 2, 8]) Batch, Ped ID, x|y, Seq Len 
        
        #Now sample 20 samples
        ade_ls = {}
        fde_ls = {}
        V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())
        V_x_rel_to_abs = nodes_rel_to_nodes_abs(V_obs.data.cpu().numpy().squeeze().copy(),
                                                 V_x[0,:,:].copy())

        V_y = seq_to_nodes(pred_traj_gt.data.cpu().numpy().copy())
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.data.cpu().numpy().squeeze().copy(),
                                                 V_x[-1,:,:].copy())
        
        raw_data_dict[step] = {}
        raw_data_dict[step]['obs'] = copy.deepcopy(V_x_rel_to_abs)
        raw_data_dict[step]['trgt'] = copy.deepcopy(V_y_rel_to_abs)
        raw_data_dict[step]['pred'] = []

        raw_data_dict[step]['frame_obs']=copy.deepcopy(frame_obs)
        raw_data_dict[step]['frame_pred']=copy.deepcopy(frame_pred)

        for n in range(num_of_objs):
            ade_ls[n]=[]
            fde_ls[n]=[]

        for k in range(KSTEPS):

            V_pred = mvnormal.sample()

            print(V_pred.shape)



            #V_pred = seq_to_nodes(pred_traj_gt.data.numpy().copy())
            V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_pred.data.cpu().numpy().squeeze().copy(),
                                                     V_x[-1,:,:].copy())
            raw_data_dict[step]['pred'].append(copy.deepcopy(V_pred_rel_to_abs))
            
           # print(V_pred_rel_to_abs.shape) #(12, 3, 2) = seq, ped, location
            for n in range(num_of_objs):
                pred = [] 
                target = []
                obsrvs = [] 
                number_of = []
                pred.append(V_pred_rel_to_abs[:,n:n+1,:])
                target.append(V_y_rel_to_abs[:,n:n+1,:])
                obsrvs.append(V_x_rel_to_abs[:,n:n+1,:])
                number_of.append(1)

                ade_ls[n].append(ade(pred,target,number_of))
                fde_ls[n].append(fde(pred,target,number_of))
        
        for n in range(num_of_objs):
            ade_bigls.append(min(ade_ls[n]))
            fde_bigls.append(min(fde_ls[n]))

    ade_ = sum(ade_bigls)/len(ade_bigls)
    fde_ = sum(fde_bigls)/len(fde_bigls)
    return ade_,fde_,raw_data_dict


paths = ['./checkpoint_social_stgcnn_transformer/tag']
# paths =["/home/asyed/Social-STGCNN/cvpr_version/Social-STGCNN/ST_transformer/checkpoint_social_stgcnn_transformer/hotel/tag"]
# paths = ["/home/asyed/Social-STGCNN/cvpr_version/Social-STGCNN/ST_transformer/checkpoint_social_stgcnn_transformer/tag"]
KSTEPS=20

print("*"*50)
print('Number of samples:',KSTEPS)
print("*"*50)




for feta in range(len(paths)):
    ade_ls = [] 
    fde_ls = [] 
    path = paths[feta]
    exps = glob.glob(path)
    print('Model being tested are:',exps)

    for exp_path in exps:
        print("*"*50)
        print("Evaluating model:",exp_path)

        model_path = exp_path+'/val_best.pth'
        args_path = exp_path+'/args.pkl'
        with open(args_path,'rb') as f: 
            args = pickle.load(f)

        stats= exp_path+'/constant_metrics.pkl'
        with open(stats,'rb') as f: 
            cm = pickle.load(f)
        print("Stats:",cm)



        #Data prep     
        obs_seq_len = args.obs_seq_len
        pred_seq_len = args.pred_seq_len
        #data_set = './datasets/'+args.dataset+'/'
       # data_set = './datasets/'+"hotel"+'/'
        data_set= "/home/asyed/Social-STGCNN/cvpr_version/Social-STGCNN/datasets/stanford/ped/"

        print("DATASET IS TO TEST IS", data_set)
        dset_test = TrajectoryDataset(
                data_set+'test',
                #data_set+'/',


                obs_len=obs_seq_len,
                pred_len=pred_seq_len,
                skip=1,norm_lap_matr=True)

        loader_test = DataLoader(
                dset_test,
                batch_size=1,#This is irrelative to the args batch size parameter
                shuffle =False,
                num_workers=1)



        #Defining the model 
        model = social_stgcnn(n_stgcnn =args.n_stgcnn,n_txpcnn=args.n_txpcnn,
        output_feat=args.output_size,seq_len=args.obs_seq_len,
        kernel_size=args.kernel_size,pred_seq_len=args.pred_seq_len).cuda()
        model.load_state_dict(torch.load(model_path))


        ade_ =999999
        fde_ =999999
        print("Testing ....")
        ad,fd,raw_data_dic_= test()
        path_traj ="/home/asyed/Social-STGCNN/cvpr_version/Social-STGCNN/ST_transformer/trajs/stanford/"
        with open(path_traj + "stanford1.pkl","wb+") as out:
              pickle.dump(raw_data_dic_,out)
        ade_= min(ade_,ad)
        fde_ =min(fde_,fd)
        ade_ls.append(ade_)
        fde_ls.append(fde_)
        print("ADE:",ade_," FDE:",fde_)




    print("*"*50)

    print("Avg ADE:",sum(ade_ls)/5)
    print("Avg FDE:",sum(fde_ls)/5)
