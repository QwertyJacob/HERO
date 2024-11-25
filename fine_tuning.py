"""
Copyright (c) 2022 Jesus Cevallos, University of Insubria

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Except as contained in this notice, the name of the University of Insubria
shall not be used in advertising or otherwise to promote the sale, use or other
dealings in this Software without prior written authorization from the
University of Insubria.

The University of Insubria retains all rights to the Software, including but not
limited to all patent rights, trade secret rights, know-how, and other
intellectual property rights.

Non-commercial use of the Software is permitted, but the User must cite the
Software in any publication or presentation that uses the Software, and must
not use the Software for commercial purposes without first obtaining the
written permission of the University of Insubria.
"""
import wandb
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange
import numpy as np
from omegaconf import DictConfig, OmegaConf 
import hydra

import NAR.utils as utils
import NAR.data_management as dm
import NAR.plotting_utils as pu
import NAR.masking as masking
import NAR.models as models

from torchvision.models import resnet18


def save_stuff(cfg):
    torch.save(
        encoder.state_dict(),
        cfg.models_dir+'/'+cfg.fine_tuning.run_name+'_encoder.pt')
    torch.save(
        decoder_1_b.state_dict(),
        cfg.models_dir+'/'+cfg.fine_tuning.run_name+'_dec_1_b.pt')
    torch.save(
        decoder_2_b.state_dict(),
        cfg.models_dir+'/'+cfg.fine_tuning.run_name+'_dec_2_b.pt')


def init_data(cfg):
    global micro_zdas, micro_type_A_ZdAs, micro_type_B_ZdAs, micro_classes, macro_classes

    data = utils.init_bot_iot_ds_from_dir(cfg.fine_tuning.datadir)
    micro_zdas = cfg.fine_tuning.curriculum.micro_zdas
    micro_type_A_ZdAs = cfg.fine_tuning.curriculum.micro_type_A_zdas
    micro_type_B_ZdAs = cfg.fine_tuning.curriculum.micro_type_B_zdas
    train_type_B_micro_classes = cfg.fine_tuning.curriculum.train_type_B_micro_classes
    test_type_B_micro_classes = cfg.fine_tuning.curriculum.test_type_B_micro_classes
    test_type_A_macro_classes = cfg.fine_tuning.curriculum.test_type_A_macro_classes
    train_type_A_macro_classes = cfg.fine_tuning.curriculum.train_type_A_macro_classes

    data = masking.mask_real_data(
        data=data,
        micro_zdas=micro_zdas,
        micro_type_A_ZdAs=micro_type_A_ZdAs,
        micro_type_B_ZdAs=micro_type_B_ZdAs
        )

    train_data, test_data = masking.split_real_data(
        data,
        train_type_B_micro_classes,
        test_type_B_micro_classes,
        test_type_A_macro_classes,
        train_type_A_macro_classes
        )

    micro_classes = data['Micro Label'].unique()
    macro_classes = data['Macro Label'].unique()

    print('Train Type A ZdAs: \n', train_data[train_data.Type_A_ZdA]['Micro Label'].unique())
    print('Train Type B ZdAs: \n', train_data[train_data.Type_B_ZdA]['Micro Label'].unique())
    print('Test Type A ZdAs: \n', test_data[test_data.Type_A_ZdA]['Micro Label'].unique())
    print('Test Type B ZdAs: \n', test_data[test_data.Type_B_ZdA]['Micro Label'].unique())

    train_dataset = dm.RealFewShotDataset(
        root_dir=cfg.fine_tuning.datadir,
        df=train_data)


    tensors = []
    labels = []

    for idx_set in tqdm(train_dataset.idxs_per_micro_class.items()):
        for single_idx in idx_set[1][:50]:
            item = train_dataset[int(single_idx)]
            img = item[0][:512*512].flatten()
            tensors.append(img)
            labels.append(item[1])

    tensors = torch.vstack(tensors)
    labels = torch.vstack(labels)


    pu.plot_hidden_space(
        mod='Train',
        hiddens=tensors,
        labels=labels,
        cat='Micro',
        nl_labels=train_dataset.micro_label_encoder.classes_)

    # the maximum number of records you want for each Micro Label
    max_records = cfg.fine_tuning.max_samples_per_class
    
    if cfg.fine_tuning.use_preset_sampling:
        reduced_train_data = pd.read_csv(f'reduced_data_mappings/reduced_train_data_{max_records}.csv')
    else:
        # Function to select the top N records for each group
        def select_top_n(group):
            return group.head(max_records)

        reduced_train_data = train_data.groupby('Micro Label', group_keys=False).apply(select_top_n)
        reduced_train_data = reduced_train_data.reset_index(drop=True)
        reduced_train_data.to_csv(f'reduced_train_data_{max_records}.csv', index=False)

    # Dataset and Dataloader:
    train_dataset = dm.RealFewShotDataset(
        root_dir=cfg.fine_tuning.datadir,
        df=reduced_train_data)

    test_dataset = dm.RealFewShotDataset(
        root_dir=cfg.fine_tuning.datadir,
        df=test_data)

    train_loader = DataLoader(
    dataset=train_dataset,
    sampler=dm.FewShotSamplerReal(
                dataset=train_dataset,
                n_tasks=cfg.fine_tuning.n_train_tasks,
                classes_per_it=cfg.fine_tuning.N_WAY,
                k_shot=cfg.fine_tuning.N_SHOT,
                q_shot=cfg.fine_tuning.N_QUERY),
    num_workers=cfg.fine_tuning.num_workers,
    drop_last=True,
    collate_fn=dm.convenient_cf)

    test_loader = DataLoader(
        dataset=test_dataset,
        sampler=dm.FewShotSamplerReal(
                    dataset=test_dataset,
                    n_tasks=cfg.fine_tuning.n_eval_tasks,
                    classes_per_it=cfg.fine_tuning.N_WAY,
                    k_shot=cfg.fine_tuning.N_SHOT,
                    q_shot=cfg.fine_tuning.N_QUERY),
        num_workers=cfg.fine_tuning.num_workers,
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
    global enc_proc_optimizer, os_optimizer, trainable_decoders

    # Encoder
    encoder = resnet18().to(device)
    encoder.fc = nn.Sequential(nn.Flatten(),
                           nn.Linear(512,
                                     cfg.fine_tuning.h_dim)).to(device)
    # First phase:
    processor_1 = models.GAT_V5_Processor(
                    h_dim=cfg.fine_tuning.h_dim,
                    processor_attention_heads=cfg.fine_tuning.processor_attention_heads,
                    dropout=cfg.fine_tuning.dropout,
                    device=device
                    ).to(device)

    decoder_1a_criterion = nn.CrossEntropyLoss()

    # You may need to change the params in the constructor to fit the decoder module.
    decoder_class = getattr(models, cfg.fine_tuning.decoder)
    decoder_1_b = decoder_class(
                    in_dim=cfg.fine_tuning.N_WAY-2, # Subtrack a type A and a type B ZdA attack
                    dropout=cfg.fine_tuning.dropout,
                    device=device
                    ).to(device)

    decoder_1b_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.Tensor([cfg.fine_tuning.pos_weight_1])).to(device)

    # Second phase:
    processor_2 = models.GAT_V5_Processor(
                    h_dim=cfg.fine_tuning.h_dim,
                    processor_attention_heads=cfg.fine_tuning.processor_attention_heads,
                    dropout=cfg.fine_tuning.dropout,
                    device=device
                    ).to(device)

    if cfg.fine_tuning.pretrained_processors:
        load_pretrained_processors_weights(cfg)

    decoder_2a_criterion = nn.CrossEntropyLoss().to(device)

    decoder_2_b = decoder_class(
                    in_dim=cfg.fine_tuning.N_WAY-1, # Only type A attack will be discarded from known realm
                    dropout=cfg.fine_tuning.dropout,
                    device=device
                    ).to(device)

    decoder_2b_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.Tensor([cfg.fine_tuning.pos_weight_2])).to(device)


    params_for_processor_optimizer = \
            list(encoder.parameters()) + \
            list(processor_1.parameters()) + \
            list(processor_2.parameters())

    enc_proc_optimizer = optim.Adam(
        params_for_processor_optimizer,
        lr=cfg.fine_tuning.lr)
    if any(param.requires_grad for param in decoder_1_b.parameters()): 
        trainable_decoders = True
        params_for_os_optimizer = \
                list(decoder_1_b.parameters()) + \
                list(decoder_2_b.parameters())

        os_optimizer = optim.Adam(
            params_for_os_optimizer,
            lr=cfg.fine_tuning.lr)
    else:
        trainable_decoders = False

def init_logging(cfg, train_loader, test_loader):
    print('Initializing logging for ', cfg.fine_tuning.run_name)

    if cfg.fine_tuning.wandb:
        wandb.login()

    finetuning_conf = dict(cfg.fine_tuning)
    additional_confs = {
                    "test_classes": test_loader.dataset.micro_classes,
                    "train_classes": train_loader.dataset.micro_classes,
                    "train_batch_size": iter(train_loader).__next__()[0].shape[1],
                    "len(train_loader)": len(train_loader),
                    "len(test_dataset)": len(test_loader.dataset),
                }

    finetuning_conf.update(additional_confs)

    wandb.init(project='HERO',
               name=cfg.fine_tuning.run_name,
               mode=("online" if wb else "disabled"),
               config=finetuning_conf)
    if cfg.fine_tuning.wandb and cfg.fine_tuning.track_gradients:
        wandb.watch(processor_1)
        wandb.watch(processor_2)
        wandb.watch(encoder)
        wandb.watch(decoder_1_b)
        wandb.watch(decoder_2_b)


def first_phase_simple(
        cfg,
        sample_batch,
        eval=False):
    global cs_cm_1, os_cm_1, metrics_dict
    global eval_zda_labels, eval_zda_preds


    # get masks: THESE ARE NOT COMPLEMENTARY!
    zda_mask, _, \
        zda_or_query_mask, \
        known_class_query_mask = utils.get_masks_1(
            sample_batch[1],
            cfg.fine_tuning.N_QUERY,
            device=device)

    # get one_hot_labels:
    micro_oh_labels = utils.get_oh_labels(
        decimal_labels=sample_batch[1][:, 1].long(),
        total_classes=cfg.fine_tuning.max_prototype_buffer_micro,
        device=device)

    # mask labels:
    oh_masked_labels = utils.get_one_hot_masked_labels(
        micro_oh_labels,
        zda_or_query_mask,
        device=device)

    # encoding input space:
    encoded_inputs = encoder(
        sample_batch[0].float())

    # processing
    logits_wrt_known_classes, hiddens_1, predicted_kernel = processor_1(
        encoded_inputs,
        oh_masked_labels)

    # semantic kernel:
    semantic_kernel = micro_oh_labels @ micro_oh_labels.T

    # Processor training loss:
    micro_kernel_loss = utils.get_kernel_kernel_loss(
        semantic_kernel,
        predicted_kernel,
        a_w=cfg.fine_tuning.attr_w,
        r_w=cfg.fine_tuning.rep_w)

    # Transform labels for Few_shot Closed-set classif.
    # compatible with the design of models.get_centroids functions,
    # wich is called by our GAT processors.
    unique_labels, transformed_labels = sample_batch[1][:, 1][known_class_query_mask].unique(
        return_inverse=True)

    # closed set classification
    micro_classification_loss = decoder_1a_criterion(
        logits_wrt_known_classes[known_class_query_mask],
        transformed_labels)

    # Detach closed from open set gradients
    input_for_os_dec = logits_wrt_known_classes.detach()
    input_for_os_dec.requires_grad = True

    # Unknown cluster prediction:
    predicted_zdas = decoder_1_b(
        scores=input_for_os_dec[zda_or_query_mask]
        )

    # open-set loss:
    zda_detection_loss = decoder_1b_criterion(
        predicted_zdas,
        zda_mask[zda_or_query_mask].float().unsqueeze(-1))

    if eval:
        eval_zda_labels.append(zda_mask[zda_or_query_mask].long())
        eval_zda_preds.append((predicted_zdas > 0.5).long())

    # inverse transform cs preds (just for reporting)
    logits_wrt_all_classes = utils.inverse_transform_preds(
        transormed_preds=logits_wrt_known_classes[known_class_query_mask],
        real_labels=unique_labels,
        real_class_num=cfg.fine_tuning.max_prototype_buffer_micro)

    #
    # REPORTING:
    #

    # Closed set confusion matrix
    cs_cm_1 += utils.efficient_cm(
        preds=logits_wrt_all_classes.detach(),
        targets=sample_batch[1][:, 1][known_class_query_mask].long())

    # Open set confusion matrix
    os_cm_1 += utils.efficient_os_cm(
        preds=(predicted_zdas.detach() > 0.5).long(),
        targets=zda_mask[zda_or_query_mask].long()
        )

    # accuracies:
    micro_classification_acc = utils.get_acc(
        logits_preds=logits_wrt_all_classes,
        oh_labels=micro_oh_labels[known_class_query_mask])

    zda_detection_accuracy = utils.get_binary_acc(
        logits=predicted_zdas.detach(),
        labels=zda_mask[zda_or_query_mask].float().unsqueeze(-1))

    epoch_zda_dect_acc = utils.get_balanced_accuracy(
                os_cm=os_cm_1,
                n_w=cfg.fine_tuning.balanced_acc_n_w
                )

    # for reporting:
    metrics_dict['losses_1a'].append(micro_classification_loss.item())
    metrics_dict['proc_reg_loss1'].append(micro_kernel_loss.item())
    metrics_dict['CS_accuracies'].append(micro_classification_acc.item())
    metrics_dict['losses_1b'].append(zda_detection_loss.item())
    metrics_dict['OS_accuracies'].append(zda_detection_accuracy.item())
    metrics_dict['OS_B_accuracies'].append(epoch_zda_dect_acc.item())

    # Processor loss:
    proc_1_loss = micro_classification_loss + micro_kernel_loss

    return proc_1_loss, \
        zda_detection_loss, \
        hiddens_1, \
        logits_wrt_known_classes


def second_phase_simple(
        cfg, 
        sample_batch,
        hiddens_1,
        eval=False):

    global cs_cm_2, os_cm_2, metrics_dict
    global eval_type_a_labels, eval_type_a_preds
    
    type_A_mask, _, \
        type_A_or_query_mask, non_type_A_query_mask = utils.get_masks_2(
            sample_batch[1],
            cfg.fine_tuning.N_QUERY,
            device=device)

    # get one_hot_labels:
    macro_oh_labels = utils.get_oh_labels(
        decimal_labels=sample_batch[1][:, 0].long(),
        total_classes=cfg.fine_tuning.max_prototype_buffer_macro,
        device=device)

    # mask labels:
    oh_masked_labels = utils.get_one_hot_masked_labels(
        macro_oh_labels,
        type_A_or_query_mask,
        device=device)

    logist_wrt_known_macro_classes, hiddens_2, predicted_kernel_2 = processor_2(
        hiddens_1,
        oh_masked_labels)

    # semantic kernel:
    semantic_kernel_2 = macro_oh_labels @ macro_oh_labels.T

    # Processor training loss:
    proc_2_reg_loss = utils.get_kernel_kernel_loss(
        semantic_kernel_2,
        predicted_kernel_2,
        a_w=cfg.fine_tuning.attr_w,
        r_w=cfg.fine_tuning.rep_w)

    unique_macro_labels, transformed_labels_2 = sample_batch[1][:, 0][non_type_A_query_mask].unique(
        return_inverse=True)

    # Closed set: should learn to associate type B's to corr. macro cluster.
    # Notice this could imply a inconsistencies in the manifold learning dynamics of the
    # first phase! (Gradients 2A)
    macro_classification_loss = decoder_2a_criterion(
        logist_wrt_known_macro_classes[non_type_A_query_mask],
        transformed_labels_2)

    input_for_os_dec_2 = logist_wrt_known_macro_classes.detach()
    input_for_os_dec_2.requires_grad = True

    # Unknown cluster prediction:
    predicted_type_As = decoder_2_b(
        scores=input_for_os_dec_2[type_A_or_query_mask]
        )

    # open-set loss:
    type_A_detect_loss = decoder_2b_criterion(
        predicted_type_As,
        type_A_mask[type_A_or_query_mask].float().unsqueeze(-1))

    if eval:
        eval_type_a_labels.append(type_A_mask[type_A_or_query_mask].float().unsqueeze(-1))
        eval_type_a_preds.append(predicted_type_As.detach())

    # inverse transform cs preds (just for reporting)
    logits_wrt_all_macro_classes = utils.inverse_transform_preds(
        transormed_preds=logist_wrt_known_macro_classes[non_type_A_query_mask],
        real_labels=unique_macro_labels,
        real_class_num=cfg.fine_tuning.max_prototype_buffer_macro)

    # Closed set confusion matrix
    cs_cm_2 += utils.efficient_cm(
        preds=logits_wrt_all_macro_classes.detach(),
        targets=sample_batch[1][:, 0][non_type_A_query_mask].long(),
        )

    # Open set confusion matrix
    os_cm_2 += utils.efficient_os_cm(
        preds=(predicted_type_As.detach() > 0.5).long(),
        targets=type_A_mask[type_A_or_query_mask].long()
        )

    # accuracies:
    macro_classif_acc = utils.get_acc(
        logits_preds=logits_wrt_all_macro_classes,
        oh_labels=macro_oh_labels[non_type_A_query_mask])

    type_A_classif_acc = utils.get_binary_acc(
        logits=predicted_type_As.detach(),
        labels=type_A_mask[type_A_or_query_mask].float().unsqueeze(-1))

    apoch_type_A_classif_acc = utils.get_balanced_accuracy(
                os_cm=os_cm_2,
                n_w=cfg.fine_tuning.balanced_acc_n_w
                )

    proc_2_loss = macro_classification_loss + proc_2_reg_loss

    # for reporting:
    metrics_dict['losses_2a'].append(macro_classification_loss.item())
    metrics_dict['proc_reg_loss2'].append(proc_2_reg_loss.item())
    metrics_dict['losses_2b'].append(type_A_detect_loss.item())
    metrics_dict['CS_2_accuracies'].append(macro_classif_acc.item())
    metrics_dict['OS_2_accuracies'].append(type_A_classif_acc.item())
    metrics_dict['OS_2_B_accuracies'].append(apoch_type_A_classif_acc.item())

    return proc_2_loss, \
        type_A_detect_loss, \
        hiddens_2, \
        logist_wrt_known_macro_classes


def load_pretrained_processors_weights(cfg):
    processor_1.load_state_dict(torch.load(cfg.models_dir+'/SufflePretrain_H1024_proc_1.pt', weights_only=True))
    processor_2.load_state_dict(torch.load(cfg.models_dir+'/SufflePretrain_H1024_proc_2.pt', weights_only=True))

    if not cfg.fine_tuning.retrain_processors:
        processor_1.eval()
        processor_2.eval()

        # Freeze the pre-trained processor
        for param in processor_1.parameters():
            param.requires_grad = False
        # Freeze the pre-trained processor
        for param in processor_2.parameters():
            param.requires_grad = False


def train(cfg, train_loader, test_loader):
    global cs_cm_1, os_cm_1, cs_cm_2, os_cm_2, metrics_dict
    global max_eval_TNR, epochs_without_improvement
    global eval_zda_labels, eval_type_a_labels, eval_zda_preds, eval_type_a_preds

    step = 0
    max_eval_TNR = torch.zeros(1, device=device)
    epochs_without_improvement = 0

    for epoch in trange(cfg.fine_tuning.n_epochs):

        # TRAIN
        encoder.train()
        decoder_1_b.train()
        decoder_2_b.train()
        if not cfg.fine_tuning.pretrained_processors or cfg.fine_tuning.retrain_processors:
            processor_1.train()
            processor_2.train()
        
        # reset conf Mats
        cs_cm_1 = torch.zeros(
            [cfg.fine_tuning.max_prototype_buffer_micro, cfg.fine_tuning.max_prototype_buffer_micro],
            device=device)
        os_cm_1 = torch.zeros([2, 2], device=device)
        cs_cm_2 = torch.zeros(
            [cfg.fine_tuning.max_prototype_buffer_macro, cfg.fine_tuning.max_prototype_buffer_macro],
            device=device)
        os_cm_2 = torch.zeros([2, 2], device=device)

        # reset metrics dict
        metrics_dict = utils.reset_metrics_dict()

        # go!
        for batch_idx, sample_batch in enumerate(train_loader):

            step += 1

            # go to cuda:
            sample_batch = sample_batch[0].to(device), sample_batch[1].to(device)

            # PHASE 1
            proc_1_loss, \
                zda_detect_loss, \
                hiddens_1, \
                micro_known_logits = first_phase_simple(
                                    cfg,
                                    sample_batch)

            # PHASE 2
            proc_2_loss, \
                type_a_detect_loss, \
                hiddens_2, \
                decoded_2 = second_phase_simple(
                    cfg,
                    sample_batch,
                    hiddens_1)

            # Learning
            if trainable_decoders:
                proc_loss = proc_1_loss + proc_2_loss
                enc_proc_optimizer.zero_grad()
                proc_loss.backward()
                enc_proc_optimizer.step()

                os_loss = zda_detect_loss + type_a_detect_loss
                os_optimizer.zero_grad()
                os_loss.backward()
                os_optimizer.step()
            else:
                proc_loss = proc_1_loss + proc_2_loss + zda_detect_loss + type_a_detect_loss
                enc_proc_optimizer.zero_grad()
                proc_loss.backward()
                enc_proc_optimizer.step()


            if step % cfg.fine_tuning.report_step_frequency == 0:
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
                    scores_1=micro_known_logits.detach().cpu(),
                    scores_2=decoded_2.detach().cpu(),
                    cs_cm_1=cs_cm_1.cpu(),
                    cs_cm_2=cs_cm_2.cpu(),
                    os_cm_1=os_cm_1.cpu(),
                    os_cm_2=os_cm_2.cpu(),
                    wb=wb,
                    wandb=wandb,
                    step=step,
                    complete_micro_classes=micro_classes,
                    complete_macro_classes=macro_classes
                    )

        with torch.inference_mode():

            # Evaluation
            encoder.eval()
            decoder_1_b.eval()
            decoder_2_b.eval()

            # reset conf Mats
            cs_cm_1 = torch.zeros(
                [cfg.fine_tuning.max_prototype_buffer_micro, cfg.fine_tuning.max_prototype_buffer_micro],
                device=device)
            os_cm_1 = torch.zeros([2, 2], device=device)
            cs_cm_2 = torch.zeros(
                [cfg.fine_tuning.max_prototype_buffer_macro, cfg.fine_tuning.max_prototype_buffer_macro],
                device=device)
            os_cm_2 = torch.zeros([2, 2], device=device)

            # reset metrics dict
            metrics_dict = utils.reset_metrics_dict()

            # reset labels and logits for ROC AUC
            eval_zda_labels = []
            eval_type_a_labels = []
            eval_zda_preds = []
            eval_type_a_preds = []

            # go!
            for batch_idx, sample_batch in enumerate(test_loader):

                step += 1

                # go to cuda:
                sample_batch = sample_batch[0].to(device), sample_batch[1].to(device)

                # PHASE 1
                proc_1_loss, \
                    zda_detect_loss, \
                    hiddens_1, \
                    micro_known_logits = first_phase_simple(
                        cfg,
                        sample_batch,
                        eval=True)

                # PHASE 2
                proc_2_loss, \
                    type_a_detect_loss, \
                    hiddens_2, \
                    decoded_2 = second_phase_simple(
                        cfg,
                        sample_batch,
                        hiddens_1,
                        eval=True)

                if step % cfg.fine_tuning.report_step_frequency == 0:

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
                    scores_1=micro_known_logits.detach().cpu(),
                    scores_2=decoded_2.detach().cpu(),
                    cs_cm_1=cs_cm_1.cpu(),
                    cs_cm_2=cs_cm_2.cpu(),
                    os_cm_1=os_cm_1.cpu(),
                    os_cm_2=os_cm_2.cpu(),
                    wb=wb,
                    wandb=wandb,
                    step=step,
                    complete_micro_classes=micro_classes,
                    complete_macro_classes=macro_classes,
                )

            # Checking for improvement
            curr_TNR = np.array(metrics_dict['OS_2_B_accuracies']).mean()

            utils.report_auc_stuff(
                eval_zda_labels,
                eval_zda_preds,
                eval_type_a_labels,
                eval_type_a_preds,
                wb,
                wandb,
                step)

            if curr_TNR > max_eval_TNR:
                max_eval_TNR = curr_TNR
                epochs_without_improvement = 0
                if cfg.fine_tuning.save_model: save_stuff(cfg)
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= cfg.fine_tuning.patience:
                print(f'Early stopping at step {step}')
                if wb:
                    wandb.log({'Early stopping at step': step}, step=step)
                break

    print(f'max_eval_TNR: {max_eval_TNR}')

    if wb:
        wandb.finish()


@hydra.main(config_path="config", config_name="default", version_base="1.2")
def main(cfg: DictConfig) -> None:
    global wb, device

    if cfg.override != "":
        try:
            # Load the variant specified from the command line
            config_overrides = OmegaConf.load(hydra.utils.get_original_cwd() + f'/config/overrides/{cfg.override}.yaml')
            # Merge configurations, with the variant overriding the base config
            cfg = OmegaConf.merge(cfg, config_overrides)
        except:
            print('Unsuccesfully tried to use the configuration override: ',cfg.override)

    print(cfg.fine_tuning)
    wb = cfg.fine_tuning.wandb
    device = cfg.fine_tuning.device
    torch.manual_seed(cfg.seed) # for reproducibility

    train_loader, test_loader = init_data(cfg)

    init_models(cfg)

    init_logging(cfg, train_loader, test_loader)

    train(cfg, train_loader, test_loader)


if __name__ == '__main__':
    main()

