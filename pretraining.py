import wandb
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm
import numpy as np
from omegaconf import DictConfig 
import hydra

import NAR.utils as utils
import NAR.data_management as dm
import NAR.plotting_utils as pu
import NAR.masking as masking
import NAR.models as models


def save_stuff(prefix):
    torch.save(
        processor_1.state_dict(),
        'models/'+prefix+'_proc_1.pt')
    torch.save(
        processor_2.state_dict(),
        'models/'+prefix+'_proc_2.pt')

def init_data(cfg):
    global micro_zdas, micro_type_A_ZdAs, micro_type_B_ZdAs, micro_classes, macro_classes

    train_data = pd.read_csv(f'{cfg.pretraining.datadir}/train_{cfg.pretraining.dataset}.csv')
    test_data = pd.read_csv(f'{cfg.pretraining.datadir}/test_{cfg.pretraining.dataset}.csv')

    data = pd.concat([train_data, test_data])
    micro_zdas = data[data.ZdA == True]['Micro Label'].unique()
    micro_type_A_ZdAs = data[data.Type_A_ZdA == True]['Micro Label'].unique()
    micro_type_B_ZdAs = data[data.Type_B_ZdA == True]['Micro Label'].unique()
    micro_classes = data['Micro Label'].unique()
    micro_classes.sort()
    macro_classes = data['Macro Label'].unique()
    macro_classes.sort()

    print('Train Type A ZdAs: \n', train_data[train_data.Type_A_ZdA]['Micro Label'].unique())
    print('Train Type B ZdAs: \n', train_data[train_data.Type_B_ZdA]['Micro Label'].unique())
    print('Test Type A ZdAs: \n', test_data[test_data.Type_A_ZdA]['Micro Label'].unique())
    print('Test Type B ZdAs: \n', test_data[test_data.Type_B_ZdA]['Micro Label'].unique())

    assert cfg.pretraining.natural_inputs_dim == int(cfg.pretraining.dataset.split('DIM_')[1].split('_')[0]), 'Declared Natural inputs dim should be less or equal than dataset DIM'       
    
    new_train_data = train_data.drop(columns=[
        'Micro Label',
        'Macro Label',
        'ZdA',
        'Type_A_ZdA',
        'Type_B_ZdA']).values[:, :cfg.pretraining.natural_inputs_dim]

    new_test_data = test_data.drop(columns=[
            'Micro Label',
            'Macro Label',
            'ZdA',
            'Type_A_ZdA',
            'Type_B_ZdA']).values[:, :cfg.pretraining.natural_inputs_dim]

    # NOISE SCHEME:

    # Simulating different noise in different samples :)
    train_samples = new_train_data.shape[0]
    test_samples = new_test_data.shape[0]

    # Distribution of noise among features:
    noise_tensor = torch.rand(1, cfg.pretraining.natural_inputs_dim) * cfg.pretraining.noise
    noise_train = noise_tensor.repeat(train_samples, 1)
    noise_test = noise_tensor.repeat(test_samples, 1)

    new_train_data = new_train_data + noise_train.numpy()
    new_test_data = new_test_data + noise_test.numpy()

    # Dataset and Dataloader:
    train_dataset = dm.SynthFewShotDataset(
        features=new_train_data,
        df=train_data)

    test_dataset = dm.SynthFewShotDataset(
        features=new_test_data,
        df=test_data)

    train_loader = DataLoader(
    dataset=train_dataset,
    sampler=dm.FewShotSampler(
                dataset=train_dataset,
                n_tasks=cfg.pretraining.n_train_tasks,
                classes_per_it=cfg.pretraining.N_WAY,
                k_shot=cfg.pretraining.N_SHOT,
                q_shot=cfg.pretraining.N_QUERY),
    num_workers=cfg.pretraining.num_workers,
    drop_last=True,
    collate_fn=dm.convenient_cf)

    test_loader = DataLoader(
        dataset=test_dataset,
        sampler=dm.FewShotSampler(
                    dataset=test_dataset,
                    n_tasks=cfg.pretraining.n_eval_tasks,
                    classes_per_it=cfg.pretraining.N_WAY,
                    k_shot=cfg.pretraining.N_SHOT,
                    q_shot=cfg.pretraining.N_QUERY),
        num_workers=cfg.pretraining.num_workers,
        drop_last=True,
        collate_fn=dm.convenient_cf)

    # reproducibility
    train_loader.sampler.reset()
    test_loader.sampler.reset()

    return train_loader, test_loader


def init_models(cfg):
    # Make these objects accessible everywhere
    global encoder, processor_1, decoder_1a_criterion, decoder_1_b, decoder_1b_criterion
    global processor_2, decoder_2a_criterion, decoder_2_b, decoder_2b_criterion
    global processor_optimizer, os_optimizer

    # Encoder
    encoder = models.Encoder(
        in_features=cfg.pretraining.natural_inputs_dim,
        out_features=cfg.pretraining.h_dim,
        norm=cfg.pretraining.norm,
        dropout=cfg.pretraining.dropout,
        ).to(cfg.pretraining.device)

    # First phase:
    processor_1 = models.GAT_V5_Processor(
                    h_dim=cfg.pretraining.h_dim,
                    processor_attention_heads=cfg.pretraining.processor_attention_heads,
                    dropout=cfg.pretraining.dropout,
                    device=cfg.pretraining.device
                    ).to(cfg.pretraining.device)

    decoder_1a_criterion = nn.CrossEntropyLoss()

    decoder_1_b = models.Confidence_Decoder(
                    in_dim=cfg.pretraining.N_WAY-2, # Subtrack a type A and a type B ZdA attack
                    dropout=cfg.pretraining.dropout,
                    device=cfg.pretraining.device
                    ).to(cfg.pretraining.device)

    decoder_1b_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.Tensor([cfg.pretraining.pos_weight_1])).to(cfg.pretraining.device)

    # Second phase:
    processor_2 = models.GAT_V5_Processor(
                    h_dim=cfg.pretraining.h_dim,
                    processor_attention_heads=cfg.pretraining.processor_attention_heads,
                    dropout=cfg.pretraining.dropout,
                    device=cfg.pretraining.device
                    ).to(cfg.pretraining.device)

    decoder_2a_criterion = nn.CrossEntropyLoss().to(cfg.pretraining.device)

    decoder_2_b = models.Confidence_Decoder(
                    in_dim=cfg.pretraining.N_WAY-1, # Only type A attack will be discarded from known realm
                    dropout=cfg.pretraining.dropout,
                    device=cfg.pretraining.device
                    ).to(cfg.pretraining.device)

    decoder_2b_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.Tensor([cfg.pretraining.pos_weight_2])).to(cfg.pretraining.device)


    params_for_processor_optimizer = \
            list(encoder.parameters()) + \
            list(processor_1.parameters()) + \
            list(processor_2.parameters())

    processor_optimizer = optim.Adam(
        params_for_processor_optimizer,
        lr=cfg.pretraining.lr)

    params_for_os_optimizer = \
            list(decoder_1_b.parameters()) + \
            list(decoder_2_b.parameters())

    os_optimizer = optim.Adam(
        params_for_os_optimizer,
        lr=cfg.pretraining.lr)


def init_logging(cfg, train_loader, test_loader):
    print('Initializing logging... for ', cfg.pretraining.run_name)
    wandb.init(project='HERO',
               name=cfg.pretraining.run_name,
               mode=("online" if wb else "disabled"),
               config={"N_SHOT": cfg.pretraining.N_SHOT,
                       "N_QUERY": cfg.pretraining.N_QUERY,
                       "N_WAY": cfg.pretraining.N_WAY,
                       "test_classes": test_loader.dataset.micro_classes,
                       "train_classes": train_loader.dataset.micro_classes,
                       "train_batch_size": iter(train_loader).__next__()[0].shape[1],
                       "len(train_loader)": len(train_loader),
                       "len(test_dataset)": len(test_loader.dataset),
                       "max_prototype_buffer_micro": cfg.pretraining.max_prototype_buffer_micro,
                       "max_prototype_buffer_macro": cfg.pretraining.max_prototype_buffer_macro,
                       "device": device,
                       "natural_inputs_dim": cfg.pretraining.natural_inputs_dim,
                       "h_dim": cfg.pretraining.h_dim,
                       "lr": cfg.pretraining.lr,
                       "n_epochs": cfg.pretraining.n_epochs,
                       "norm": cfg.pretraining.norm,
                       "dropout": cfg.pretraining.dropout,
                       "patience": cfg.pretraining.patience,
                       'micro_zdas': micro_zdas,
                       'micro_type_A_ZdAs': micro_type_A_ZdAs,
                       'micro_type_B_ZdAs': micro_type_B_ZdAs,
                       "lambda_os": cfg.pretraining.lambda_os,
                       "positive_weight_1": cfg.pretraining.pos_weight_1,
                       "positive_weight_2": cfg.pretraining.pos_weight_2,
                       "balanced_acc_n_w": cfg.pretraining.balanced_acc_n_w,
                       "attr_w": cfg.pretraining.attr_w,
                       "rep_w": cfg.pretraining.rep_w
                       })


def first_phase_simple(
        cfg,
        sample_batch):
    global cs_cm_1
    global os_cm_1
    global metrics_dict

    # get masks: THESE ARE NOT COMPLEMENTARY!
    zda_mask, _, \
        unknown_1_mask, \
        active_query_mask = utils.get_masks_1(
            sample_batch[1],
            cfg.pretraining.N_QUERY,
            device=device)

    # get one_hot_labels:
    oh_labels = utils.get_oh_labels(
        decimal_labels=sample_batch[1][:, 1].long(),
        total_classes=cfg.pretraining.max_prototype_buffer_micro,
        device=device)

    # mask labels:
    oh_masked_labels = utils.get_one_hot_masked_labels(
        oh_labels,
        unknown_1_mask,
        device=device)

    # encoding input space:
    encoded_inputs = encoder(
        sample_batch[0].float())

    # processing
    decoded_1, hiddens_1, predicted_kernel = processor_1(
        encoded_inputs,
        oh_masked_labels)

    # semantic kernel:
    semantic_kernel = oh_labels @ oh_labels.T
    # Processor regularization:
    proc_1_reg_loss = utils.get_kernel_kernel_loss(
        semantic_kernel,
        predicted_kernel,
        a_w=cfg.pretraining.attr_w,
        r_w=cfg.pretraining.rep_w)

    # Transform labels for Few_shot Closed-set classif.
    # compatible with the design of models.get_centroids functions,
    # wich is called by our GAT processors.
    unique_labels, transformed_labels = sample_batch[1][:, 1][active_query_mask].unique(
        return_inverse=True)

    # closed set classification
    dec_1_loss_a = decoder_1a_criterion(
        decoded_1[active_query_mask],
        transformed_labels)

    # Detach closed from open set gradients
    input_for_os_dec = decoded_1.detach()
    input_for_os_dec.requires_grad = True

    # Unknown cluster prediction:
    predicted_unknown_1s = decoder_1_b(
        scores=input_for_os_dec[unknown_1_mask]
        )

    # open-set loss:
    dec_1_loss_b = decoder_1b_criterion(
        predicted_unknown_1s,
        zda_mask[unknown_1_mask].float().unsqueeze(-1))

    # inverse transform cs preds
    it_preds = utils.inverse_transform_preds(
        transormed_preds=decoded_1[active_query_mask],
        real_labels=unique_labels,
        real_class_num=cfg.pretraining.max_prototype_buffer_micro)

    #
    # REPORTING:
    #

    # Closed set confusion matrix
    cs_cm_1 += utils.efficient_cm(
        preds=it_preds.detach(),
        targets=sample_batch[1][:, 1][active_query_mask].long())

    # Open set confusion matrix
    os_cm_1 += utils.efficient_os_cm(
        preds=(predicted_unknown_1s.detach() > 0.5).long(),
        targets=zda_mask[unknown_1_mask].long()
        )

    # accuracies:
    CS_acc = utils.get_acc(
        logits_preds=it_preds,
        oh_labels=oh_labels[active_query_mask])

    OS_acc = utils.get_binary_acc(
        logits=predicted_unknown_1s.detach(),
        labels=zda_mask[unknown_1_mask].float().unsqueeze(-1))

    OS_b_acc = utils.get_balanced_accuracy(
                os_cm=os_cm_1,
                n_w=cfg.pretraining.balanced_acc_n_w
                )

    # for reporting:
    metrics_dict['losses_1a'].append(dec_1_loss_a.item())
    metrics_dict['proc_reg_loss1'].append(proc_1_reg_loss.item())
    metrics_dict['CS_accuracies'].append(CS_acc.item())
    metrics_dict['losses_1b'].append(dec_1_loss_b.item())
    metrics_dict['OS_accuracies'].append(OS_acc.item())
    metrics_dict['OS_B_accuracies'].append(OS_b_acc.item())

    # Processor loss:
    proc_1_loss = dec_1_loss_a + proc_1_reg_loss

    return proc_1_loss, \
        dec_1_loss_b, \
        hiddens_1, \
        decoded_1


def second_phase_simple(
        cfg, 
        sample_batch,
        hiddens_1):

    global cs_cm_2
    global os_cm_2
    global metrics_dict

    # get masks: THESE ARE NOT COMPLEMETARY!
    type_A_mask, known_macro_classes_mask, \
        unknown_2_mask, active_query_mask_2 = utils.get_masks_2(
            sample_batch[1],
            cfg.pretraining.N_QUERY,
            device=device)

    # get one_hot_labels:
    oh_labels = utils.get_oh_labels(
        decimal_labels=sample_batch[1][:, 0].long(),
        total_classes=cfg.pretraining.max_prototype_buffer_macro,
        device=device)

    # mask labels:
    oh_masked_labels = utils.get_one_hot_masked_labels(
        oh_labels,
        unknown_2_mask,
        device=device)

    decoded_2, hiddens_2, predicted_kernel_2 = processor_2(
        hiddens_1,
        oh_masked_labels)

    # semantic kernel:
    semantic_kernel_2 = oh_labels @ oh_labels.T
    # Processor regularization:
    proc_2_reg_loss = utils.get_kernel_kernel_loss(
        semantic_kernel_2,
        predicted_kernel_2,
        a_w=cfg.pretraining.attr_w,
        r_w=cfg.pretraining.rep_w)

    unique_macro_labels, transformed_labels_2 = sample_batch[1][:, 0][active_query_mask_2].unique(
        return_inverse=True)

    # Closed set: should learn to associate type B's to corr. macro cluster.
    # geometrical "break" in the real-data case. (GRadients 2A)
    dec_2_loss_a = decoder_2a_criterion(
        decoded_2[active_query_mask_2],
        transformed_labels_2)

    input_for_os_dec_2 = decoded_2.detach()
    input_for_os_dec_2.requires_grad = True

    # Unknown cluster prediction:
    predicted_unknown_2s = decoder_2_b(
        scores=input_for_os_dec_2[unknown_2_mask]
        )

    # open-set loss:
    dec_2_loss_b = decoder_2b_criterion(
        predicted_unknown_2s,
        type_A_mask[unknown_2_mask].float().unsqueeze(-1))

    # inverse transform cs preds
    it_preds = utils.inverse_transform_preds(
        transormed_preds=decoded_2[active_query_mask_2],
        real_labels=unique_macro_labels,
        real_class_num=cfg.pretraining.max_prototype_buffer_macro)

    # Closed set confusion matrix
    cs_cm_2 += utils.efficient_cm(
        preds=it_preds.detach(),
        targets=sample_batch[1][:, 0][active_query_mask_2].long(),
        )

    # Open set confusion matrix
    os_cm_2 += utils.efficient_os_cm(
        preds=(predicted_unknown_2s.detach() > 0.5).long(),
        targets=type_A_mask[unknown_2_mask].long()
        )

    # accuracies:
    CS_acc_2 = utils.get_acc(
        logits_preds=it_preds,
        oh_labels=oh_labels[active_query_mask_2])

    OS_acc_2 = utils.get_binary_acc(
        logits=predicted_unknown_2s.detach(),
        labels=type_A_mask[unknown_2_mask].float().unsqueeze(-1))

    OS_2_B_acc = utils.get_balanced_accuracy(
                os_cm=os_cm_2,
                n_w=cfg.pretraining.balanced_acc_n_w
                )

    proc_2_loss = dec_2_loss_a + proc_2_reg_loss

    # for reporting:
    metrics_dict['losses_2a'].append(dec_2_loss_a.item())
    metrics_dict['proc_reg_loss2'].append(proc_2_reg_loss.item())
    metrics_dict['losses_2b'].append(dec_2_loss_b.item())
    metrics_dict['CS_2_accuracies'].append(CS_acc_2.item())
    metrics_dict['OS_2_accuracies'].append(OS_acc_2.item())
    metrics_dict['OS_2_B_accuracies'].append(OS_2_B_acc.item())

    return proc_2_loss, \
        dec_2_loss_b, \
        hiddens_2, \
        decoded_2


def pretrain(cfg, train_loader, test_loader):
    global cs_cm_1, os_cm_1, cs_cm_2, os_cm_2, metrics_dict

    if wb:
        wandb.login()
        wandb.watch(processor_1)
        wandb.watch(processor_2)
        wandb.watch(encoder)
        wandb.watch(decoder_1_b)
        wandb.watch(decoder_2_b)

    for epoch in tqdm(range(cfg.pretraining.n_epochs)):

        # TRAIN
        encoder.train()
        processor_1.train()
        decoder_1_b.train()
        processor_2.train()
        decoder_2_b.train()

        # reset conf Mats
        cs_cm_1 = torch.zeros(
            [cfg.pretraining.max_prototype_buffer_micro, cfg.pretraining.max_prototype_buffer_micro],
            device=device)
        os_cm_1 = torch.zeros([2, 2], device=device)
        cs_cm_2 = torch.zeros(
            [cfg.pretraining.max_prototype_buffer_macro, cfg.pretraining.max_prototype_buffer_macro],
            device=device)
        os_cm_2 = torch.zeros([2, 2], device=device)

        # reset metrics dict
        metrics_dict = utils.reset_metrics_dict()

        # go!
        for batch_idx, sample_batch in enumerate(train_loader):
            # go to cuda:
            sample_batch = sample_batch[0].to(device), sample_batch[1].to(device)

            # PHASE 1
            proc_1_loss, \
                os_1_loss, \
                hiddens_1, \
                decoded_1 = first_phase_simple(
                                    cfg,
                                    sample_batch)

            # PHASE 2
            proc_2_loss, \
                os_2_loss, \
                hiddens_2, \
                decoded_2 = second_phase_simple(
                    cfg,
                    sample_batch,
                    hiddens_1)

            # Learning
            proc_loss = proc_1_loss + proc_2_loss
            processor_optimizer.zero_grad()
            proc_loss.backward()
            processor_optimizer.step()

            os_loss = os_1_loss + os_2_loss
            os_optimizer.zero_grad()
            os_loss.backward()
            os_optimizer.step()

            # Reporting
            step = batch_idx + (epoch * train_loader.sampler.n_tasks)

            if step % cfg.pretraining.report_step_frequency == 0:
                utils.reporting_simple(
                    'train',
                    epoch,
                    metrics_dict,
                    step,
                    wb,
                    wandb)

        pu.super_plotting_function(
                    phase='Training',
                    labels=sample_batch[1].cpu(),
                    hiddens_1=hiddens_1.detach().cpu(),
                    hiddens_2=hiddens_2.detach().cpu(),
                    scores_1=decoded_1.detach().cpu(),
                    scores_2=decoded_2.detach().cpu(),
                    cs_cm_1=cs_cm_1.cpu(),
                    cs_cm_2=cs_cm_2.cpu(),
                    os_cm_1=os_cm_1.cpu(),
                    os_cm_2=os_cm_2.cpu(),
                    wb=wb,
                    wandb=wandb,
                    complete_micro_classes=micro_classes,
                    complete_macro_classes=macro_classes
                    )

        with torch.inference_mode():

            # Evaluation
            encoder.eval()
            processor_1.eval()
            decoder_1_b.eval()
            processor_2.eval()
            decoder_2_b.eval()

            # reset conf Mats
            cs_cm_1 = torch.zeros(
                [cfg.pretraining.max_prototype_buffer_micro, cfg.pretraining.max_prototype_buffer_micro],
                device=device)
            os_cm_1 = torch.zeros([2, 2], device=device)
            cs_cm_2 = torch.zeros(
                [cfg.pretraining.max_prototype_buffer_macro, cfg.pretraining.max_prototype_buffer_macro],
                device=device)
            os_cm_2 = torch.zeros([2, 2], device=device)

            # reset metrics dict
            metrics_dict = utils.reset_metrics_dict()

            # go!
            for batch_idx, sample_batch in enumerate(test_loader):
                # go to cuda:
                sample_batch = sample_batch[0].to(device), sample_batch[1].to(device)

                # PHASE 1
                proc_1_loss, \
                    os_1_loss, \
                    hiddens_1, \
                    decoded_1 = first_phase_simple(
                        cfg,
                        sample_batch)

                # PHASE 2
                proc_2_loss, \
                    os_2_loss, \
                    hiddens_2, \
                    decoded_2 = second_phase_simple(
                        cfg,
                        sample_batch,
                        hiddens_1)

                # Reporting
                step = batch_idx + (epoch * test_loader.sampler.n_tasks)

                if step % cfg.pretraining.report_step_frequency == 0:
                    utils.reporting_simple(
                            'eval',
                            epoch,
                            metrics_dict,
                            step,
                            wb,
                            wandb)

            pu.super_plotting_function(
                    phase='Evaluation',
                    labels=sample_batch[1].cpu(),
                    hiddens_1=hiddens_1.detach().cpu(),
                    hiddens_2=hiddens_2.detach().cpu(),
                    scores_1=decoded_1.detach().cpu(),
                    scores_2=decoded_2.detach().cpu(),
                    cs_cm_1=cs_cm_1.cpu(),
                    cs_cm_2=cs_cm_2.cpu(),
                    os_cm_1=os_cm_1.cpu(),
                    os_cm_2=os_cm_2.cpu(),
                    wb=wb,
                    wandb=wandb,
                    complete_micro_classes=micro_classes,
                    complete_macro_classes=macro_classes
                )

            # Checking for improvement
            curr_TNR = np.array(metrics_dict['OS_2_B_accuracies']).mean()

            if curr_TNR > max_eval_TNR:
                max_eval_TNR = curr_TNR
                epochs_without_improvement = 0
                if cfg.pretraining.save_model: save_stuff(run_name)
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= cfg.pretraining.patience:
                print(f'Early stopping at step {step}')
                if wb:
                    wandb.log({'Early stopping at episode': step})
                break

    print(f'max_eval_TNR: {max_eval_TNR}')

    if wb:
        wandb.finish()



@hydra.main(config_path="config", config_name="default", version_base="1.2")
def main(cfg: DictConfig) -> None:
    global wb, device
    print(cfg)
    wb = cfg.pretraining.wandb
    device = cfg.pretraining.device
    torch.manual_seed(cfg.seed) # for reproducibility

    train_loader, test_loader = init_data(cfg)

    init_models(cfg)

    init_logging(cfg, train_loader, test_loader)

    pretrain(cfg, train_loader, test_loader)


if __name__ == '__main__':
    main()

