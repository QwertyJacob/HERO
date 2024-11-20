import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
import itertools

"""
PLOTTING STUFF
"""

zda_dict = {
    0.0: 'Known attack',
    1.0: 'Type B ZdA',
    2.0: 'Type A ZdA'
}

# List of colors
colors = [
    'red', 'blue', 'green', 'purple', 'orange', 'pink', 'cyan',  'brown', 'yellow',
    'olive', 'lime', 'teal', 'maroon', 'navy', 'fuchsia', 'aqua', 'silver', 'sienna', 'gold',
    'indigo', 'violet', 'turquoise', 'tomato', 'orchid', 'slategray', 'peru', 'magenta', 'limegreen',
    'royalblue', 'coral', 'darkorange', 'darkviolet', 'darkslateblue', 'dodgerblue', 'firebrick',
    'lightseagreen', 'mediumorchid', 'orangered', 'powderblue', 'seagreen', 'springgreen', 'tan', 'wheat',
    'burlywood', 'chartreuse', 'crimson', 'darkgoldenrod', 'darkolivegreen', 'darkseagreen', 'indianred',
    'lavender', 'lightcoral', 'lightpink', 'lightsalmon', 'limegreen', 'mediumseagreen', 'mediumpurple',
    'midnightblue', 'palegreen', 'rosybrown', 'saddlebrown', 'salmon', 'slateblue', 'steelblue',
]




def plot_hidden_space(
        mod,
        hiddens,
        labels,
        cat='Macro',
        nl_labels=None,
        wb=False,
        wandb=None):

    # Create an iterator that cycles through the colors
    color_iterator = itertools.cycle(colors)

    # If dimensionality is > 2, reduce using PCA
    if hiddens.shape[1]>2:
        pca = PCA(n_components=2)
        hiddens = pca.fit_transform(hiddens)

    plt.figure(figsize=(10, 6))

    # Two plots:
    plt.subplot(1, 1, 1)
    
    # Choose correct labels:
    if cat == 'Macro':
        p_labels = labels[:, 0]
    else:
        p_labels = labels[:, 1]

    # natural language labels?
    if nl_labels is not None:
        p_labels = nl_labels[p_labels.long()]

    # ZdA Labels:
    type_a_labels = 2 * labels[:, 2]            # type As
    type_b_labels = labels[:, 3]                # type Bs
    zda_labels = type_a_labels + type_b_labels

    # List of attacks:
    unique_p_labels = np.unique(p_labels)

    # Print points for each attack
    for label in unique_p_labels:
        data = hiddens[p_labels == label]

        label_for_scatter = f'{cat}_{label}'
        color_for_scatter = next(color_iterator)

        if labels[:, 2][p_labels == label][0] == True:
            label_for_scatter += '\n (Type A ZdA)'
            color_for_scatter = 'black'
        
        if cat == 'Micro':
            if labels[:, 3][p_labels == label][0] == True:
                label_for_scatter += '\n (Type B ZdA)'
                color_for_scatter = 'grey'

        plt.scatter(
            data[:, 0],
            data[:, 1],
            label=label_for_scatter,
            c=color_for_scatter)
            
    plt.title(f'{mod}: Latent Space of {cat} attacks')

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()

    if wb:
        wandb.log({f"{mod}: Latent Space of {cat} attacks": wandb.Image(plt)})
    else:
        plt.show()

    plt.cla()
    plt.close()




def plot_hidden_space_gennaro(
        mod,
        hiddens,
        labels,
        nl_labels=None,
        wb=False,
        wandb=None):

    # Create an iterator that cycles through the colors
    color_iterator = itertools.cycle(colors)

    # If dimensionality is > 2, reduce using PCA
    if hiddens.shape[1]>2:
        pca = PCA(n_components=2)
        hiddens = pca.fit_transform(hiddens)

    plt.figure(figsize=(10, 6))

    # Two plots:
    plt.subplot(1, 1, 1)
        
    p_labels = labels[:, 0]

    # natural language labels?
    if nl_labels is not None:
        p_labels = nl_labels[p_labels.long()]

    # ZdA Labels:
    zda_labels = labels[:, 1]

    # List of attacks:
    unique_p_labels = np.unique(p_labels)

    # Print points for each attack
    for label in unique_p_labels:
        data = hiddens[p_labels == label]

        label_for_scatter = f'Cluster {label}'
        color_for_scatter = next(color_iterator)

        if labels[:, 1][p_labels == label][0] == True:
            label_for_scatter += '\n (ZdA)'
            color_for_scatter = 'black'

        plt.scatter(
            data[:, 0],
            data[:, 1],
            label=label_for_scatter,
            c=color_for_scatter)
            
    plt.title(f'{mod}: Latent Space Representations')

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()

    if wb:
        wandb.log({f"{mod}: Latent Space Representations": wandb.Image(plt)})
    else:
        plt.show()

    plt.cla()
    plt.close()


def plot_scores_reduction(
        mod,
        hiddens,
        labels,
        cat='Macro',
        nl_labels=None,
        wb=False,
        wandb=None):

    # Create an iterator that cycles through the colors
    color_iterator = itertools.cycle(colors)
    
    pca = PCA(n_components=2)
    hiddens = pca.fit_transform(hiddens)

    plt.figure(figsize=(10, 6))

    # Two plots:
    plt.subplot(1, 1, 1)
    
    # Choose correct labels:
    if cat == 'Macro':
        p_labels = labels[:, 0]
    else:
        p_labels = labels[:, 1]

    # natural language labels?
    if nl_labels is not None:
        p_labels = nl_labels[p_labels.long()]

    # ZdA Labels:
    type_a_labels = 2 * labels[:, 2]            # type As
    type_b_labels = labels[:, 3]                # type Bs
    zda_labels = type_a_labels + type_b_labels

    # List of attacks:
    unique_p_labels = np.unique(p_labels)

    # Print points for each attack
    for label in unique_p_labels:
        data = hiddens[p_labels == label]

        label_for_scatter = f'{cat}_{label}'
        color_for_scatter = next(color_iterator)

        if labels[:, 2][p_labels == label][0] == True:
            label_for_scatter += '\n (Type A ZdA)'
            color_for_scatter = 'black'
        
        if cat == 'Micro':
            if labels[:, 3][p_labels == label][0] == True:
                label_for_scatter += '\n (Type B ZdA)'
                color_for_scatter = 'grey'

        plt.scatter(
            data[:, 0],
            data[:, 1],
            label=label_for_scatter,
            c=color_for_scatter)
            
    plt.title(f'{mod}: PCA reduction of {cat} association scores')

    if len(unique_p_labels) < 20:
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))


    plt.tight_layout()

    if wb:
        wandb.log({f"{mod}: PCA of {cat} ass. scores": wandb.Image(plt)})
    else:
        plt.show()

    plt.cla()
    plt.close()


def plot_scores_reduction_gennaro(
        mod,
        hiddens,
        labels,
        nl_labels=None,
        wb=False,
        wandb=None):

    # Create an iterator that cycles through the colors
    color_iterator = itertools.cycle(colors)
    
    pca = PCA(n_components=2)
    hiddens = pca.fit_transform(hiddens)

    plt.figure(figsize=(10, 6))

    # Two plots:
    plt.subplot(1, 1, 1)
    
    # Choose correct labels:
    p_labels = labels[:, 0]

    # natural language labels?
    if nl_labels is not None:
        p_labels = nl_labels[p_labels.long()]

    # ZdA Labels:
    zda_labels = labels[:, 1] 

    # List of attacks:
    unique_p_labels = np.unique(p_labels)

    # Print points for each attack
    for label in unique_p_labels:
        data = hiddens[p_labels == label]

        label_for_scatter = f'Cluster {label}'
        color_for_scatter = next(color_iterator)

        if labels[:, 1][p_labels == label][0] == True:
            label_for_scatter += '\n (ZdA)'
            color_for_scatter = 'black'

        plt.scatter(
            data[:, 0],
            data[:, 1],
            label=label_for_scatter,
            c=color_for_scatter)
            
    plt.title(f'{mod}: PCA reduction of association scores')

    if len(unique_p_labels) < 20:
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))


    plt.tight_layout()

    if wb:
        wandb.log({f"{mod}: PCA of ass. scores": wandb.Image(plt)})
    else:
        plt.show()

    plt.cla()
    plt.close()


def plot_confusion_matrix(
        cm,
        mode,
        phase,
        wb=False,
        wandb=None,
        norm=True,
        dims=(10,10),
        classes=None):

    if norm:
        # Rapresented classes:
        rep_classes = cm.sum(1) > 0
        # Normalize
        denom = cm.sum(1).reshape(-1, 1)
        denom[~rep_classes] = 1
        cm = cm / denom
        fmt_str = ".2f"
    else:
        fmt_str = ".0f"

    # Plot heatmap using seaborn
    sns.set()
    plt.figure(figsize=dims)
    ax = sns.heatmap(
        cm,
        annot=True,
        cmap='Blues',
        fmt=fmt_str,
        xticklabels=classes, 
        yticklabels=classes)

    # Rotate x-axis and y-axis labels vertically
    ax.set_xticklabels(classes, rotation=90)
    ax.set_yticklabels(classes, rotation=0)

    # Add x and y axis labels
    plt.xlabel("Predicted")
    plt.ylabel("Baseline")

    plt.title(f'{phase} {mode} set Confusion Matrix')
    if wb:
        wandb.log({f'{phase} {mode} set Confusion Matrix': wandb.Image(plt)})
    else:
        plt.show()
    plt.cla()
    plt.close()


def plot_dataset(data):
    # Separate data into macro and micro labels
    macro_data = data[data['Macro Label'].notnull()]
    micro_data = data[data['Micro Label'].notnull()]

    # Create separate plots for macro labels and micro labels
    plt.figure(figsize=(12, 6))

    # Plot for macro labels
    plt.subplot(1, 2, 1)
    for macro_label in macro_data['Macro Label'].unique():
        macro_label_data = macro_data[macro_data['Macro Label'] == macro_label]
        plt.scatter(macro_label_data['0'], macro_label_data['1'], label=f'Macro {macro_label}')
    plt.title('Macro Labels')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()

    # Plot for micro labels
    plt.subplot(1, 2, 2)
    for micro_label in micro_data['Micro Label'].unique():
        micro_label_data = micro_data[micro_data['Micro Label'] == micro_label]
        plt.scatter(micro_label_data['0'], micro_label_data['1'], label=f'Micro {micro_label}')
    plt.title('Micro Labels')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    plt.tight_layout()
    plt.show()


def plot_masked_dataset(
    data, type='Micro'):

    micro_type_A_ZdAs = data[data['Type_A_ZdA'] == True]['Micro Label'].unique()
    micro_type_B_ZdAs = data[data['Type_B_ZdA'] == True]['Micro Label'].unique()
    macro_type_A_ZdAs = data[data['Type_A_ZdA'] == True]['Macro Label'].unique()

    # Separate data into macro and micro labels
    macro_data = data[data['Macro Label'].notnull()]
    micro_data = data[data['Micro Label'].notnull()]

    # Create separate plots for macro labels and micro labels
    plt.figure(figsize=(20, 10))


    # Plot for micro labels
    plt.subplot(1, 2, 1)
    for micro_label in micro_data['Micro Label'].unique():
        micro_label_data = micro_data[micro_data['Micro Label'] == micro_label]
        if np.isin(micro_label, micro_type_B_ZdAs):
            plt.scatter(micro_label_data['0'],
                        micro_label_data['1'],
                        label=f'Type B ZdA {micro_label}',
                        c='gray',
                        alpha=0.3,
                        s=2)
        elif np.isin(micro_label, micro_type_A_ZdAs):
            plt.scatter(micro_label_data['0'],
                        micro_label_data['1'],
                        label=f'Type A ZdA {micro_label}',
                        c='black',
                        alpha=0.3,
                        s=2)
        else:
            plt.scatter(micro_label_data['0'],
                        micro_label_data['1'],
                        label=f'Micro {micro_label}',
                        alpha=0.5,
                        s=2)

    plt.title('Synthetic Data')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.subplot(1, 2, 2)
    for macro_label in macro_data['Macro Label'].unique():
        macro_label_data = macro_data[macro_data['Macro Label'] == macro_label]
        
        if np.isin(macro_label, macro_type_A_ZdAs):
            plt.scatter(macro_label_data['0'],
                        macro_label_data['1'],
                        label=f'Type A ZdA {macro_label}',
                        c='black',
                        alpha=0.3,
                        s=2)
        else:
            plt.scatter(macro_label_data['0'],
                        macro_label_data['1'],
                        label=f'Macro {macro_label}',
                        alpha=0.5,
                        s=2)

    plt.title('Synthetic Data')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()


def super_plotting_function(
        phase,
        labels,
        hiddens_1,
        hiddens_2,
        scores_1,
        scores_2,
        cs_cm_1,
        cs_cm_2,
        os_cm_1,
        os_cm_2,
        wb,
        wandb,
        nl_micro_labels=None,
        nl_macro_labels=None,
        complete_micro_classes=None,
        complete_macro_classes=None
        ):

    plot_hidden_space(
            mod=phase,
            hiddens=hiddens_1,
            labels=labels,
            nl_labels=nl_micro_labels,
            cat='Micro',
            wb=wb,
            wandb=wandb)

    plot_scores_reduction(
        mod=phase,
        hiddens=scores_1,
        labels=labels,
        nl_labels=nl_micro_labels,
        cat='Micro',
        wb=wb,
        wandb=wandb)

    plot_hidden_space(
        mod=phase,
        hiddens=hiddens_2,
        labels=labels,
        nl_labels=nl_macro_labels,
        cat='Macro',
        wb=wb,
        wandb=wandb)

    plot_scores_reduction(
        mod=phase,
        hiddens=scores_2,
        labels=labels,
        nl_labels=nl_macro_labels,
        cat='Macro',
        wb=wb,
        wandb=wandb)

    os_labels_micro = ['Known', 'Unknown']
    os_labels_macro = ['Type B ZdA', 'Type A ZdA']

    plot_confusion_matrix(
            cm=cs_cm_1,
            mode='(Micro) Closed',
            phase=phase,
            wb=wb,
            wandb=wandb,
            norm=False,
            dims=(15,15),
            classes=complete_micro_classes)

    plot_confusion_matrix(
            cm=os_cm_1,
            mode='(Micro) Open',
            phase=phase,
            wb=wb,
            wandb=wandb,
            norm=False,
            dims=(4,4),
            classes=os_labels_micro)

    plot_confusion_matrix(
            cm=cs_cm_2,
            mode='(Macro) Closed',
            phase=phase,
            wb=wb,
            wandb=wandb,
            norm=False,
            dims=(4,4),
            classes=complete_macro_classes)

    plot_confusion_matrix(
            cm=os_cm_2,
            mode='(Macro) Open',
            phase=phase,
            wb=wb,
            wandb=wandb,
            norm=False,
            dims=(4,4),
            classes=os_labels_macro)



def super_plotting_function_gennaro(
        phase,
        labels,
        hiddens_1,
        scores_1,
        cs_cm_1,
        os_cm_1,
        wb,
        wandb,
        nl_labels=None,
        complete_classes=None,
        ):

    plot_hidden_space_gennaro(
            mod=phase,
            hiddens=hiddens_1,
            labels=labels,
            nl_labels=nl_labels,
            wb=wb,
            wandb=wandb)

    plot_scores_reduction_gennaro(
        mod=phase,
        hiddens=scores_1,
        labels=labels,
        nl_labels=nl_labels,
        wb=wb,
        wandb=wandb)

    os_labels_micro = ['Known Attack', 'ZdA']

    plot_confusion_matrix(
            cm=cs_cm_1,
            mode='Closed',
            phase=phase,
            wb=wb,
            wandb=wandb,
            norm=False,
            dims=(15,15),
            classes=complete_classes)

    plot_confusion_matrix(
            cm=os_cm_1,
            mode='Open',
            phase=phase,
            wb=wb,
            wandb=wandb,
            norm=False,
            dims=(4,4),
            classes=os_labels_micro)