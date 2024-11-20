import os
import pandas as pd
import torch
import numpy as np


def init_bot_iot_ds_from_dir(root_dir):
    micro_labels = []
    macro_labels = []
    file_names = []

    for fn in os.listdir(root_dir):
        file_names.append(fn)
        if 'OS' in fn:
            micro_labels.append('Scan_OS')
            macro_labels.append('Scan')
        elif 'Key' in fn:
            micro_labels.append('Theft_Keylogging')
            macro_labels.append('Theft')
        elif 'Serv' in fn:
            micro_labels.append('Scan_Service')
            macro_labels.append('Scan')
        elif 'Data' in fn:
            micro_labels.append('Data_Exfiltration')
            macro_labels.append('Theft')
        # ferrag's IIOTset integration!
        elif 'MITM' in fn:
            micro_labels.append('MITM')
            macro_labels.append('MITM')
        elif ('XSS' in fn) or ('SQLInjection' in fn) or ('Upl' in fn):
            micro_labels.append(fn.split('_')[0])
            macro_labels.append('Injection')
        elif ('Back' in fn) or ('Pass' in fn) or ('Ransom' in fn):
            micro_labels.append(fn.split('_')[0])
            macro_labels.append('Malware')
        else:
            micro_labels.append(fn.split('_')[0]+'_'+fn.split('_')[1])
            macro_labels.append(fn.split('_')[0])

    data = {
        'filename': file_names,
        'Micro Label': micro_labels,
        'Macro Label': macro_labels
    }

    # Create a Pandas DataFrame
    data = pd.DataFrame(data)

    return data


def init_bot_iot_gennaro(root_dir):
    labels = []
    file_names = []

    for fn in os.listdir(root_dir):
        file_names.append(fn)
        if 'OS' in fn:
            labels.append('Scan_OS')
        elif 'Key' in fn:
            labels.append('Theft_Keylogging')
        elif 'Serv' in fn:
            labels.append('Scan_Service')
        elif 'Data' in fn:
            labels.append('Data_Exfiltration')
        # ferrag's IIOTset integration!
        elif 'MITM' in fn:
            labels.append('MITM')
        elif ('XSS' in fn) or ('SQLInjection' in fn) or ('Upl' in fn) or ('Back' in fn) or ('Pass' in fn) or ('Ransom' in fn):
            labels.append(fn.split('_')[0])
        else:
            labels.append(fn.split('_')[0]+'_'+fn.split('_')[1])

    data = {
        'filename': file_names,
        'Label': labels,
    }

    # Create a Pandas DataFrame
    data = pd.DataFrame(data)

    return data


def get_normalized_adjacency_matrix(adj):
    '''
    This code first computes the degree matrix by summing the rows of the input
    adjacency matrix (adj). It then computes the inverse of the diagonal
    of the degree matrix, which is used to normalize the adjacency matrix
    element-wise. Finally, it computes the normalized adjacency matrix by
    multiplying deg_inv with adj on both sides.

    Note that this code assumes an undirected graph and uses the symmetric
    normalization approach, where the degree matrix is raised to the power
    of -0.5 and placed on both sides of the adjacency matrix.

    We don't create self loops with 1 (nor with any value)
    because we want the embeddings to adaptively learn
    the self-loop weights.

    Self-looped version (GCN):
    W = torch.eye(adj.shape[0]).cuda() + adj
    degree = torch.sum(W, dim=1).pow(-0.5)
    return (W * degree).t()*degree
    '''
    # compute degree matrix (sum of each row)
    deg = torch.sum(adj, dim=1)

    # diagonal degree matrix
    deg = torch.diag(deg)

    # compute inverse of diagonal degree matrix
    deg_inv = torch.pow(deg, -0.5)

    # numerical stability
    deg_inv[deg_inv == float('inf')] = 0

    # compute normalized adjacency matrix
    adj = deg_inv @ adj @ deg_inv

    # symmetrize
    adj = (adj + adj.t()) / 2

    return adj


def get_oh_labels(
        decimal_labels,
        total_classes,
        device):

    # create placeholder for one_hot encoding:
    labels_onehot = torch.zeros(
        [decimal_labels.size()[0],
        total_classes], device=device)
    # transform to one_hot encoding:
    labels_onehot = labels_onehot.scatter(
        1,
        decimal_labels.unsqueeze(-1),
        1)
    return labels_onehot


def get_FSL_mask(labels, N_QUERY, device):
    """
    Supposing a perfectly balanced batch, where samples are
    annotated in a round-robin fashion, we implement episodic
    training (masking label of query samples) by zeroing the last
    N_QUERY * number_of_classes rows of our label matrix:

    NOTE: Assumes the balancing classes are micro classes.
    """
    # query mask:
    # NOTE: Assumes the balancing classes are micro classes.
    balancing_labels = labels[:, 1].long()
    num_of_classes = len(balancing_labels.unique())
    query_mask = torch.zeros(
        balancing_labels.size()[0], device=device)
    query_mask[-N_QUERY * num_of_classes:] = torch.ones_like(
        query_mask[-N_QUERY * num_of_classes:], device=device)

    return query_mask

def get_gennaro_FSL_mask(labels, N_QUERY, device):
    """
    The unique change wrt to the method above is the index of 
    the used class label
    """
    # query mask:
    # NOTE: Assumes the balancing classes are micro classes.
    balancing_labels = labels[:, 0].long()
    num_of_classes = len(balancing_labels.unique())
    query_mask = torch.zeros(
        balancing_labels.size()[0], device=device)
    query_mask[-N_QUERY * num_of_classes:] = torch.ones_like(
        query_mask[-N_QUERY * num_of_classes:], device=device)

    return query_mask


def get_one_hot_masked_labels(
        one_hot_labels,
        unknown_mask,
        device='cpu'):
    # compute one_hot_masked_cluster_labels:
    ohm_cluster_labels = \
        torch.zeros_like(one_hot_labels, device=device)
    ohm_cluster_labels[~unknown_mask] = \
        one_hot_labels[~unknown_mask]
    return ohm_cluster_labels


def get_acc(logits_preds, oh_labels):

    match_mask = logits_preds.max(1)[1] == oh_labels.max(1)[1]
    return match_mask.sum() / match_mask.shape[0]


def get_binary_acc(logits, labels):

    match_mask = labels == (logits > 0.5)
    return match_mask.sum() / match_mask.shape[0]


def get_masks_1(
    labels, N_QUERY, device='cpu'):
    """
    Returns:
        zda_mask -> vertical auto-explicative_mask

        known_classes_mask -> HORIZONTAL MASK, it indicates
        which classes are not ZdAs,helps
        to evaluate the closed set prediction accuracy.

        unknown_1_mask -> VERTICAL_MASK, it indicates the ZdA
        INSTANCES and the QUERY INSTANCES in the batch.

        active_query_mask -> VERTICAL_MASK, it indicates the
        QUERY INSTANCES in the batch that are used to evaluate
        the accuracy of our CLOSED SET CLASSIFICATION.
    """
    # ZdA masks:
    type_A_mask = labels[:, 2]
    type_B_mask = labels[:, 3]
    zda_mask = torch.logical_or(
        type_A_mask,
        type_B_mask)

    # known samples mask:
    known_classes_mask = \
        labels[~zda_mask, 1].unique().long()

    # query mask:
    query_mask = get_FSL_mask(labels, N_QUERY, device)

    # active query mask:
    active_query_mask = torch.logical_and(
        query_mask,
        ~zda_mask)

    # final mask:
    unknown_1_mask = torch.logical_or(
        zda_mask,
        query_mask)

    return zda_mask, known_classes_mask, unknown_1_mask, active_query_mask




def get_gennaro_masks(
    labels, N_QUERY, device='cpu'):
    """
    Returns:
        zda_mask -> vertical auto-explicative_mask

        known_classes_mask -> HORIZONTAL MASK, it indicates
        which classes are not ZdAs,helps
        to evaluate the closed set prediction accuracy.

        unknown_1_mask -> VERTICAL_MASK, it indicates the ZdA
        INSTANCES and the QUERY INSTANCES in the batch.

        active_query_mask -> VERTICAL_MASK, it indicates the
        QUERY INSTANCES in the batch that are used to evaluate
        the accuracy of our CLOSED SET CLASSIFICATION.
    """
    # ZdA masks:
    zda_mask = labels[:, 1].bool()

    # known samples mask:
    known_classes_mask = \
        labels[~zda_mask, 1].unique().long()

    # query mask:
    query_mask = get_gennaro_FSL_mask(labels, N_QUERY, device)

    # active query mask:
    active_query_mask = torch.logical_and(
        query_mask,
        ~zda_mask)

    # final mask:
    unknown_1_mask = torch.logical_or(
        zda_mask,
        query_mask)

    return zda_mask, known_classes_mask, unknown_1_mask, active_query_mask



def get_masks_2(
    labels, N_QUERY, device='cpu'):
    """
    Returns:
        type_A_mask -> vertical auto-explicative_mask

        known_macro_classes_mask -> HORIZONTAL MASK, it indicates
        which MACRO classes are not type A ZdAs, helps
        to evaluate the closed set prediction accuracy.

        unknown_2_mask -> VERTICAL_MASK, it indicates the Type A ZdA
        INSTANCES and the QUERY INSTANCES in the batch.

        active_query_mask -> VERTICAL_MASK, it indicates the
        QUERY INSTANCES in the batch that are used to evaluate
        the accuracy of our CLOSED SET CLASSIFICATION.
    """
    # ZdA masks:
    type_A_mask = labels[:, 2].bool()

    # known samples mask:
    known_macro_classes_mask = \
        labels[~type_A_mask, 0].unique().long()

    # query mask:        
    query_mask = get_FSL_mask(labels, N_QUERY, device)

    # active query mask:
    active_query_mask = torch.logical_and(
        query_mask,
        ~type_A_mask)

    # final mask:
    unknown_2_mask = torch.logical_or(
        type_A_mask,
        query_mask)

    return type_A_mask, \
        known_macro_classes_mask, \
        unknown_2_mask, \
        active_query_mask


def reporting_simple(
        suffix,
        epoch,
        metrics_dict,
        batch_idx,
        wb,
        wandb):

    if wb:
        wandb.log(
            {'epoch: ': epoch,
             'step: ': batch_idx})
        wandb.log(
            {f'mean dec_1_loss_a_{suffix}: ':
             np.array(metrics_dict['losses_1a']).mean(),
             'step: ': batch_idx})
        wandb.log(
            {f'mean proc_reg_loss1 {suffix}: ':
             np.array(metrics_dict['proc_reg_loss1']).mean(),
             'step: ': batch_idx})
        wandb.log(
            {f'mean dec_1_loss_b_{suffix}: ':
             np.array(metrics_dict['losses_1b']).mean(),
             'step: ': batch_idx})
        wandb.log(
            {f'CS1 accuracy_{suffix}: ':
             np.array(metrics_dict['CS_accuracies']).mean(),
             'step: ': batch_idx})
        wandb.log(
            {f'OS1 accuracy_{suffix}: ':
             np.array(metrics_dict['OS_accuracies']).mean(),
             'step: ': batch_idx})
        wandb.log(
            {f'OS1 Bal. accuracy_{suffix}: ':
             np.array(metrics_dict['OS_B_accuracies']).mean(),
             'step: ': batch_idx})
        wandb.log(
            {f'mean dec_2_loss_a_{suffix}: ':
             np.array(metrics_dict['losses_2a']).mean(),
             'step: ': batch_idx})
        wandb.log(
            {f'mean proc_reg_loss2 {suffix}: ':
             np.array(metrics_dict['proc_reg_loss2']).mean(),
             'step: ': batch_idx})
        wandb.log(
            {f'mean dec_2_loss_b_{suffix}: ':
             np.array(metrics_dict['losses_2b']).mean(),
             'step: ': batch_idx})
        wandb.log(
            {f'CS2 accuracy_{suffix}: ':
             np.array(metrics_dict['CS_2_accuracies']).mean(),
             'step: ': batch_idx})
        wandb.log(
            {f'OS2 accuracy_{suffix}: ':
             np.array(metrics_dict['OS_2_accuracies']).mean(),
             'step: ': batch_idx})
        wandb.log(
            {f'OS2 Bal. accuracy_{suffix}: ':
             np.array(metrics_dict['OS_2_B_accuracies']).mean(),
             'step: ': batch_idx})

    else:
        # print(f'mean dec_1_loss_a_{suffix}: ', np.array(losses_1a).mean())
        # print(f'mean dec_1_loss_b_{suffix}: ', np.array(losses_1b).mean())
        print(f'CS accuracy_{suffix}: ',
              np.array(metrics_dict['CS_accuracies']).mean())
        print(f'OS accuracy_{suffix}: ',
              np.array(metrics_dict['OS_B_accuracies']).mean())
        # print(f'mean dec_2_loss_a_{suffix}: ', np.array(losses_2a).mean())
        # print(f'mean dec_2_loss_b_{suffix}: ', np.array(losses_2b).mean())
        print(f'CS2 accuracy_{suffix}: ',
              np.array(metrics_dict['CS_2_accuracies']).mean())
        print(f'OS2 accuracy_{suffix}: ',
              np.array(metrics_dict['OS_2_B_accuracies']).mean())


def reporting_simple_optimized(
        suffix,
        epoch,
        metrics_dict,
        batch_idx,
        report_frequency,
        wb,
        wandb):

    if batch_idx-report_frequency < 0:
        init_idx = 0
    else:
        init_idx = batch_idx-report_frequency
    if wb:
        wandb.log(
            {'epoch: ': epoch,
             'step: ': batch_idx})
        wandb.log(
            {f'mean dec_1_loss_a_{suffix}: ':
             metrics_dict['losses_1a'][init_idx:batch_idx+1].mean().item(),
             'step: ': batch_idx})
        wandb.log(
            {f'mean proc_reg_loss1 {suffix}: ':
             metrics_dict['proc_reg_loss1'][init_idx:batch_idx+1].mean().item(),
             'step: ': batch_idx})
        wandb.log(
            {f'mean dec_1_loss_b_{suffix}: ':
             metrics_dict['losses_1b'][init_idx:batch_idx+1].mean().item(),
             'step: ': batch_idx})
        wandb.log(
            {f'CS1 accuracy_{suffix}: ':
             metrics_dict['CS_accuracies'][init_idx:batch_idx+1].mean().item(),
             'step: ': batch_idx})
        wandb.log(
            {f'OS1 accuracy_{suffix}: ':
             metrics_dict['OS_accuracies'][init_idx:batch_idx+1].mean().item(),
             'step: ': batch_idx})
        wandb.log(
            {f'OS1 Bal. accuracy_{suffix}: ':
             metrics_dict['OS_B_accuracies'][init_idx:batch_idx+1].mean().item(),
             'step: ': batch_idx})
        wandb.log(
            {f'mean dec_2_loss_a_{suffix}: ':
             metrics_dict['losses_2a'][init_idx:batch_idx+1].mean().item(),
             'step: ': batch_idx})
        wandb.log(
            {f'mean proc_reg_loss2 {suffix}: ':
             metrics_dict['proc_reg_loss2'][init_idx:batch_idx+1].mean().item(),
             'step: ': batch_idx})
        wandb.log(
            {f'mean dec_2_loss_b_{suffix}: ':
             metrics_dict['losses_2b'][init_idx:batch_idx+1].mean().item(),
             'step: ': batch_idx})
        wandb.log(
            {f'CS2 accuracy_{suffix}: ':
             metrics_dict['CS_2_accuracies'][init_idx:batch_idx+1].mean().item(),
             'step: ': batch_idx})
        wandb.log(
            {f'OS2 accuracy_{suffix}: ':
             metrics_dict['OS_2_accuracies'][init_idx:batch_idx+1].mean().item(),
             'step: ': batch_idx})
        wandb.log(
            {f'OS2 Bal. accuracy_{suffix}: ':
             metrics_dict['OS_2_B_accuracies'][init_idx:batch_idx+1].mean().item(),
             'step: ': batch_idx})

    else:
        # print(f'mean dec_1_loss_a_{suffix}: ', losses_1a.mean().item())
        # print(f'mean dec_1_loss_b_{suffix}: ', losses_1b.mean().item())
        print(f'CS accuracy_{suffix}: ',
              metrics_dict['CS_accuracies'][init_idx:batch_idx+1].mean().item())
        print(f'OS accuracy_{suffix}: ',
              metrics_dict['OS_B_accuracies'][init_idx:batch_idx+1].mean().item())
        # print(f'mean dec_2_loss_a_{suffix}: ', losses_2a.mean().item())
        # print(f'mean dec_2_loss_b_{suffix}: ', losses_2b.mean().item())
        print(f'CS2 accuracy_{suffix}: ',
              metrics_dict['CS_2_accuracies'][init_idx:batch_idx+1].mean().item())
        print(f'OS2 accuracy_{suffix}: ',
              metrics_dict['OS_2_B_accuracies'][init_idx:batch_idx+1].mean().item())


def reporting_gennaro(
        suffix,
        epoch,
        metrics_dict,
        batch_idx,
        wb,
        wandb):

    if wb:
        wandb.log(
            {'epoch: ': epoch,
             'step: ': batch_idx})
        wandb.log(
            {f'mean dec_1_loss_a_{suffix}: ':
             np.array(metrics_dict['losses_1a']).mean(),
             'step: ': batch_idx})
        wandb.log(
            {f'mean proc_reg_loss1 {suffix}: ':
             np.array(metrics_dict['proc_reg_loss1']).mean(),
             'step: ': batch_idx})
        wandb.log(
            {f'mean dec_1_loss_b_{suffix}: ':
             np.array(metrics_dict['losses_1b']).mean(),
             'step: ': batch_idx})
        wandb.log(
            {f'CS1 accuracy_{suffix}: ':
             np.array(metrics_dict['CS_accuracies']).mean(),
             'step: ': batch_idx})
        wandb.log(
            {f'OS1 accuracy_{suffix}: ':
             np.array(metrics_dict['OS_accuracies']).mean(),
             'step: ': batch_idx})
        wandb.log(
            {f'OS1 Bal. accuracy_{suffix}: ':
             np.array(metrics_dict['OS_B_accuracies']).mean(),
             'step: ': batch_idx})
       
    else:
        # print(f'mean dec_1_loss_a_{suffix}: ', np.array(losses_1a).mean())
        # print(f'mean dec_1_loss_b_{suffix}: ', np.array(losses_1b).mean())
        print(f'CS accuracy_{suffix}: ',
              np.array(metrics_dict['CS_accuracies']).mean())
        print(f'OS accuracy_{suffix}: ',
              np.array(metrics_dict['OS_accuracies']).mean())


def update_benchmark(max_eval_acc,
                     epochs_without_improvement,
                     metrics_dict,
                     save):

    current_eval_acc = np.array(metrics_dict['CS_2_accuracies']).mean()

    # Checking for improvement
    if current_eval_acc > max_eval_acc:
        max_eval_acc = current_eval_acc
        epochs_without_improvement = 0
        if save:
            torch.save(
                processor_1.state_dict(),
                '../../../models/NAR_for_ZdA/processor_1.pt')

            torch.save(
                processor_2.state_dict(),
                '../../../models/NAR_for_ZdA/processor_2.pt')

            torch.save(
                decoder_1_b.state_dict(),
                '../../../models/NAR_for_ZdA/decoder_1_b.pt')

            torch.save(
                decoder_2_b.state_dict(),
                '../../../models/NAR_for_ZdA/decoder_2_b.pt')
    else:
        epochs_without_improvement += 1

    return max_eval_acc, epochs_without_improvement


def check_early_stop_simple():
    if epochs_without_improvement >= patience:
        print(f'Early stopping at epoch {epoch}')
        if wb:
            wandb.log({'Early stopping at epoch': epoch})
        return True

    return False


def reset_metrics_dict():
    return {
        'losses_1a': [],
        'losses_1b': [],
        'proc_reg_loss1': [],
        'proc_reg_loss2': [],
        'OS_accuracies': [],
        'OS_B_accuracies': [],
        'CS_accuracies': [],
        'losses_2a': [],
        'losses_2b': [],
        'CS_2_accuracies': [],
        'OS_2_accuracies': [],
        'OS_2_B_accuracies': []
    }


def reset_metrics_dict_optimized(metric_lens, device):
    return {
        'losses_1a': torch.zeros(size=(metric_lens,), device=device),
        'losses_1b': torch.zeros(size=(metric_lens,), device=device),
        'proc_reg_loss1': torch.zeros(size=(metric_lens,), device=device),
        'proc_reg_loss2': torch.zeros(size=(metric_lens,), device=device),
        'OS_accuracies': torch.zeros(size=(metric_lens,), device=device),
        'OS_B_accuracies': torch.zeros(size=(metric_lens,), device=device),
        'CS_accuracies': torch.zeros(size=(metric_lens,), device=device),
        'losses_2a': torch.zeros(size=(metric_lens,), device=device),
        'losses_2b': torch.zeros(size=(metric_lens,), device=device),
        'CS_2_accuracies': torch.zeros(size=(metric_lens,), device=device),
        'OS_2_accuracies': torch.zeros(size=(metric_lens,), device=device),
        'OS_2_B_accuracies': torch.zeros(size=(metric_lens,), device=device),
    }

def reset_metrics_dict_gennaro():
    return {
        'losses_1a': [],
        'losses_1b': [],
        'proc_reg_loss1': [],
        'OS_accuracies': [],
        'CS_accuracies': [],
        'OS_B_accuracies': []
    }


def efficient_cm(preds, targets):

    predictions_decimal = preds.argmax(dim=1).to(torch.int64)
    predictions_onehot = torch.zeros_like(
        preds,
        device=preds.device)
    predictions_onehot.scatter_(1, predictions_decimal.view(-1, 1), 1)

    targets = targets.to(torch.int64)
    # Create a one-hot encoding of the targets.
    targets_onehot = torch.zeros_like(
        preds,
        device=targets.device)
    targets_onehot.scatter_(1, targets.view(-1, 1), 1)

    return targets_onehot.T @ predictions_onehot


def efficient_os_cm(preds, targets):

    predictions_onehot = torch.zeros(
        [preds.size(0), 2],
        device=preds.device)
    predictions_onehot.scatter_(1, preds.view(-1, 1), 1)

    targets = targets.to(torch.int64)
    # Create a one-hot encoding of the targets.
    targets_onehot = torch.zeros_like(
        predictions_onehot,
        device=targets.device)
    targets_onehot.scatter_(1, targets.view(-1, 1), 1)

    return targets_onehot.T @ predictions_onehot


def get_balanced_accuracy(os_cm, n_w):
    
    N = os_cm[1][1] + os_cm[1][0]
    TN = os_cm[1][1]
    TNR = TN / N

    P = os_cm[0][0] + os_cm[0][1]
    TP = os_cm[0][0]
    TPR = TP / P
    
    return (n_w * TNR) + ((1-n_w) * TPR)


def get_kernel_loss(baseline_kernel, hiddens):
    # Reconstructed kernel:
    recons = torch.cdist(hiddens, hiddens)
    recons = torch.softmax(-recons + 1e-10, dim=1)

    # REPULSIVE force
    repulsive_CE_term = -(1 - baseline_kernel) * torch.log(1-recons + 1e-10)
    repulsive_CE_term = repulsive_CE_term.sum(dim=1)
    repulsive_CE_term = repulsive_CE_term.mean()

    # The following acts as an ATTRACTIVE force for the embedding learning:
    attractive_CE_term = -(baseline_kernel * torch.log(recons + 1e-10))
    attractive_CE_term = attractive_CE_term.sum(dim=1)
    attractive_CE_term = attractive_CE_term.mean()

    return repulsive_CE_term + attractive_CE_term


def get_kernel_kernel_loss(baseline_kernel, predicted_kernel, a_w=1, r_w=1):
    # REPULSIVE force
    repulsive_CE_term = -(1 - baseline_kernel) * torch.log(1-predicted_kernel + 1e-10)
    repulsive_CE_term = repulsive_CE_term.sum(dim=1)
    repulsive_CE_term = repulsive_CE_term.mean()

    # The following acts as an ATTRACTIVE force for the embedding learning:
    attractive_CE_term = -(baseline_kernel * torch.log(predicted_kernel + 1e-10))
    attractive_CE_term = attractive_CE_term.sum(dim=1)
    attractive_CE_term = attractive_CE_term.mean()

    return (r_w * repulsive_CE_term) + (a_w * attractive_CE_term)


def inverse_transform_preds(
    transormed_preds,
    real_labels,
    real_class_num
    ):
    """
    In FSL, we output only a little number of classes,
    and labels were also transformed. 
    We return to the absolute pred-label relationship for the 
    confusion matrix assembly
    """

    N = transormed_preds.shape[0]
    
    pos_indicator = real_labels.unsqueeze(0).expand(
        N,-1).long()
    
    it_predictions = torch.zeros(
        (N, real_class_num),
        device=transormed_preds.device).to(torch.float32)
    
    
    it_predictions = torch.scatter(
        input=it_predictions,
        dim=1, 
        index=pos_indicator, 
        src=transormed_preds)

    assert torch.all(transormed_preds == it_predictions[:,real_labels.long()])
    
    return it_predictions

