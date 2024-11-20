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
        prefix+'_proc_1.pt')
    torch.save(
        processor_2.state_dict(),
        prefix+'_proc_2.pt')

def first_phase_simple(
        sample_batch):

    global cs_cm_1
    global os_cm_1
    global metrics_dict

    # get masks: THESE ARE NOT COMPLEMETARY!
    zda_mask, \
        known_classes_mask, \
        unknown_1_mask, \
        active_query_mask = utils.get_masks_1(
            sample_batch[1],
            N_QUERY,
            device=device)

    # get one_hot_labels:
    oh_labels = utils.get_oh_labels(
        decimal_labels=sample_batch[1][:, 1].long(),
        total_classes=max_prototype_buffer_micro,
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
        a_w=attr_w,
        r_w=rep_w)

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
        real_class_num=max_prototype_buffer_micro)

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
                n_w=balanced_acc_n_w
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
        sample_batch,
        hiddens_1):

    global cs_cm_2
    global os_cm_2
    global metrics_dict

    # get masks: THESE ARE NOT COMPLEMETARY!
    type_A_mask, known_macro_classes_mask, \
        unknown_2_mask, active_query_mask_2 = utils.get_masks_2(
            sample_batch[1],
            N_QUERY,
            device=device)

    # get one_hot_labels:
    oh_labels = utils.get_oh_labels(
        decimal_labels=sample_batch[1][:, 0].long(),
        total_classes=max_prototype_buffer_macro,
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
        a_w=attr_w,
        r_w=rep_w)

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
        real_class_num=max_prototype_buffer_macro)

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
                n_w=balanced_acc_n_w
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


@hydra.main(config_path="config", config_name="default", version_base="1.2")
def main(cfg: DictConfig) -> None:

    print(cfg)

if __name__ == '__main__':
    main()

