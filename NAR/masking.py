import numpy as np
import torch

def mask_synth_data(
    data,
    n_masked_macros,         # how many macro classes do we want to mask?
    n_total_masked_micros,   # how many micro classes do we want to mask?
    ):

    # Inferring data attributes:
    M = data['Macro Label'].max() + 1
    m = (data['Micro Label'].max() + 1) // M

    macro_classes = data['Macro Label'].unique()
    micro_classes = data['Micro Label'].unique()
    macro_classes.sort()
    micro_classes.sort()


    macro_masked = n_masked_macros  # how many macro classes do we want to mask?
    assert macro_masked < M // 2
    assert macro_masked % 2 == 0  # for train test split purposes
    micro_masked = n_total_masked_micros  # how many micro classes do we want to mask?
    assert (macro_masked * m) < micro_masked < M * m


    # how many type a and type b micro zdas we will have?
    n_type_A_ZdA = (macro_masked * m)
    n_type_B_ZdA = micro_masked - n_type_A_ZdA


    # type B ZdAs might be less or equal than  (M - macro_masked -1 ) * m
    # to avoid choosing type B ZdAs that cover a whole macro category,
    # (otherwise those would be type A ZdAs by definition)
    assert n_type_B_ZdA <= (M - macro_masked - 1) * m

    # Choose random macro clusters to be type A ZdAs:
    macro_type_A_zdas = np.random.choice(
        macro_classes,
        size=macro_masked,
        replace=False)

    # Find the correspondent micro clusters:
    micro_type_A_ZdAs = data[data['Macro Label'].isin(
            macro_type_A_zdas)]['Micro Label'].unique()

    # What are the known macro categories?
    macro_clusters_left = set(macro_classes) - set(macro_type_A_zdas)

    # What are the potential known attacks + type B ZdAs?
    micro_clusters_left = list(set(micro_classes) - set(micro_type_A_ZdAs))

    # Choose type B ZdAS from the latter list
    # This code ensures not all the micro classes of a current
    # known macro category are choosen as ZdAs, because in that case,
    # those will be type A ZdAs by definition...
    micro_type_B_ZdAs = []

    for sample_iteration in range(n_type_B_ZdA):
        macro_idx = sample_iteration % len(macro_clusters_left)
        macro_cluster = list(macro_clusters_left)[macro_idx]
        micro_candidates = np.arange((macro_cluster * m),
                                     m + (macro_cluster * m))
        chosen = np.random.choice(
            micro_candidates,
            size=1,
            replace=False)
        micro_type_B_ZdAs.append(chosen[0])

    # Convenience indicator of Type B and Type A ZdAS.
    micro_zdas = list(micro_type_A_ZdAs) + list(micro_type_B_ZdAs)

    data['ZdA'] = np.where(
        data['Micro Label'].isin(micro_zdas),
        True,
        False)

    data['Type_A_ZdA'] = np.where(
        data['Micro Label'].isin(micro_type_A_ZdAs),
        True,
        False)

    data['Type_B_ZdA'] = np.where(
        data['Micro Label'].isin(micro_type_B_ZdAs),
        True,
        False)

    assert np.all(
        set(micro_type_A_ZdAs) == set(data[data.Type_A_ZdA]['Micro Label'].unique()))

    assert np.all(
        set(micro_type_B_ZdAs) == set(data[data.Type_B_ZdA]['Micro Label'].unique()))
    
    return data


def mask_real_data_lowdim(
    data,
    micro_zdas,
    micro_type_A_ZdAs,
    micro_type_B_ZdAs
    ):

    data['ZdA'] = np.where(
        data['Micro Label'].isin(micro_zdas),
        True,
        False)

    data['Type_A_ZdA'] = np.where(
        data['Micro Label'].isin(micro_type_A_ZdAs),
        True,
        False)

    data['Type_B_ZdA'] = np.where(
        data['Micro Label'].isin(micro_type_B_ZdAs),
        True,
        False)

    assert np.all(
        set(micro_type_A_ZdAs) == set(data[data.Type_A_ZdA]['Micro Label'].unique()))
    
    assert np.all(
        set(micro_type_B_ZdAs) == set(data[data.Type_B_ZdA]['Micro Label'].unique()))
    
    return data


def mask_real_data(
    data,
    micro_zdas,
    micro_type_A_ZdAs,
    micro_type_B_ZdAs
    ):

    data['ZdA'] = np.where(
        data['Micro Label'].isin(micro_zdas),
        True,
        False)

    data['Type_A_ZdA'] = np.where(
        data['Micro Label'].isin(micro_type_A_ZdAs),
        True,
        False)

    data['Type_B_ZdA'] = np.where(
        data['Micro Label'].isin(micro_type_B_ZdAs),
        True,
        False)

    data = data[[
        'filename', 'Macro Label',
        'Micro Label', 'ZdA',
        'Type_A_ZdA', 'Type_B_ZdA']]

    assert np.all(
        set(micro_type_A_ZdAs) == set(data[data.Type_A_ZdA]['Micro Label'].unique()))

    assert np.all(
        set(micro_type_B_ZdAs) == set(data[data.Type_B_ZdA]['Micro Label'].unique()))
    
    return data



def mask_real_data_gennaro(
    data,
    zdas,
    ):

    data['ZdA'] = np.where(
        data['Label'].isin(zdas),
        True,
        False)

    data = data[[
        'filename', 'Label', 'ZdA']]

    assert np.all(
        set(zdas) == set(data[data.ZdA]['Label'].unique()))

    return data


def mask_generic_data(
        data,
        zdas,
        ):

    data['ZdA'] = np.where(
        data['Label'].isin(zdas),
        True,
        False)

    assert np.all(
        set(zdas) == set(data[data.ZdA]['Label'].unique()))

    return data


def split_syth_data(
    data, 
    train_percentage=0.8):

    micro_classes = data['Micro Label'].unique()

    type_A_ZdAs = data[data.Type_A_ZdA == True]
    micro_type_A_ZdAs = type_A_ZdAs['Micro Label'].unique()
    macro_type_A_zdas = type_A_ZdAs['Macro Label'].unique()

    type_B_ZdAs = data[data.Type_B_ZdA == True]
    micro_type_B_ZdAs = type_B_ZdAs['Micro Label'].unique()

    print(f'Micro classes {micro_classes}')
    print(f'Micro type A ZdAs {micro_type_A_ZdAs}')
    print(f'Micro type B ZdAs {micro_type_B_ZdAs}')

    known_classes = np.array(
        list(
            set(micro_classes) -
            set(micro_type_B_ZdAs) -
            set(micro_type_A_ZdAs)))
    print(f'Known micro classes {known_classes}')

    # Type A ZdAs:
    print(f'Macro ZdAs (type A): {macro_type_A_zdas}')
    test_type_A_macro_classes = np.random.choice(
        macro_type_A_zdas,
        len(macro_type_A_zdas)//2)
    print(f'Test Macro ZdAs (type A): {test_type_A_macro_classes}')

    train_type_A_macro_classes = np.array(
        list(
            set(macro_type_A_zdas) -
            set(test_type_A_macro_classes)))
    print(f'Train Macro ZdAs (type A): {train_type_A_macro_classes}')

    # Type B ZdAs:
    test_type_B_micro_classes = np.random.choice(
        micro_type_B_ZdAs,
        len(micro_type_B_ZdAs)//2)
    print(f'Test type-B ZdAs: {test_type_B_micro_classes}')

    train_type_B_micro_classes = np.array(
        list(
            set(micro_type_B_ZdAs) -
            set(test_type_B_micro_classes)))
    print(f'Train type-B ZdAs: {train_type_B_micro_classes}')

    # Data splitting:
    test_mask = np.zeros(len(data))
    # some unknown attacks are exclusively in the test dataset:
    # type B:
    test_mask = np.logical_or(
        test_mask,
        data['Micro Label'].isin(test_type_B_micro_classes))
    # type A:
    test_mask = np.logical_or(
        test_mask,
        data['Macro Label'].isin(test_type_A_macro_classes))
    
    # we take a percentage of known attacks also in the test dataset:
    known_test_mask = np.zeros(len(data[~data.ZdA]))
    known_test_mask = torch.rand(len(known_test_mask)) > train_percentage
    known_test_mask = known_test_mask.numpy()
    test_mask[~data.ZdA] = known_test_mask
    # effective split:
    test_data = data[test_mask].copy()
    train_data = data[~test_mask].copy()

    return train_data, test_data


def split_real_data(
    data,
    train_type_B_micro_classes,
    test_type_B_micro_classes,
    test_type_A_macro_classes,
    train_type_A_macro_classes):
    
    macro_classes = data['Macro Label'].unique()
    micro_classes = data['Micro Label'].unique()
    macro_classes.sort()
    micro_classes.sort()
    
    type_A_ZdAs = data[data.Type_A_ZdA == True]
    micro_type_A_ZdAs = type_A_ZdAs['Micro Label'].unique()
    macro_type_A_zdas = type_A_ZdAs['Macro Label'].unique()

    assert all(item in macro_type_A_zdas for item in train_type_A_macro_classes)
    assert all(item in macro_type_A_zdas for item in test_type_A_macro_classes)
    
    type_B_ZdAs = data[data.Type_B_ZdA == True]
    micro_type_B_ZdAs = type_B_ZdAs['Micro Label'].unique()

    assert all(item in micro_type_B_ZdAs for item in train_type_B_micro_classes)
    assert all(item in micro_type_B_ZdAs for item in test_type_B_micro_classes)


    print(f'Micro classes {micro_classes}\n')
    print(f'Micro type A ZdAs {micro_type_A_ZdAs}\n')
    print(f'Micro type B ZdAs {micro_type_B_ZdAs}\n\n')

    known_classes = np.array(
        list(
            set(micro_classes) -
            set(micro_type_B_ZdAs) -
            set(micro_type_A_ZdAs)))
    print(f'Known micro classes {known_classes}\n')


    # Type A ZdAs:
    print(f'test_type_A_macro_classes {test_type_A_macro_classes}\n')
    print(f'train_type_A_macro_classes {train_type_A_macro_classes}\n')

    # Type B ZdAs:
    print(f'test_type_B_micro_classes {test_type_B_micro_classes}\n')
    print(f'train_type_B_micro_classes {train_type_B_micro_classes}\n')

    # Data splitting:
    test_mask = np.zeros(len(data))

    # some unknown attacks are exclusively in the test dataset:
    # type B:
    test_mask = np.logical_or(
        test_mask,
        data['Micro Label'].isin(test_type_B_micro_classes))
    # type A:
    test_mask = np.logical_or(
        test_mask,
        data['Macro Label'].isin(test_type_A_macro_classes))

    # we take a percentage of known attacks also in the test dataset:
    train_percentage = 0.8
    known_test_mask = np.zeros(len(data[~data.ZdA]))
    known_test_mask = torch.rand(len(known_test_mask)) > train_percentage
    known_test_mask = known_test_mask.numpy()
    test_mask[~data.ZdA] = known_test_mask
    # effective split:
    test_data = data[test_mask].copy()
    train_data = data[~test_mask].copy()
    
    return train_data, test_data


def split_real_data_gen_zda(
    data,
    train_zdas,
    test_zdas,
    test_only_knowns):
    
    classes = data['Label'].unique()
    classes.sort()
    
    ZdAs = data[data.ZdA == True]
    ZdAs = ZdAs['Label'].unique()

    assert all(item in ZdAs for item in train_zdas)
    assert all(item in ZdAs for item in test_zdas)

    print(f'Classes {classes}\n')
    print(f'ZdAs {ZdAs}\n')

    known_classes = np.array(
        list(
            set(classes) -
            set(ZdAs)))

    print(f'Known classes {known_classes}\n')


    # ZdAs:
    print(f'train_zdas {train_zdas}\n')
    print(f'test_zdas {test_zdas}\n')

    # Data splitting:
    test_mask = np.zeros(len(data))

    # some unknown attacks are exclusively in the test dataset:
    test_mask = np.logical_or(
        test_mask,
        data['Label'].isin(test_zdas))

    # some known attacks are also exclusively in the test dataset:
    test_mask = np.logical_or(
        test_mask,
        data['Label'].isin(test_only_knowns))

    # we make a classic split in the rest of data:
    train_percentage = 0.8
    known_test_mask = np.zeros(len(data[~test_mask]))
    known_test_mask = torch.rand(len(known_test_mask)) > train_percentage
    known_test_mask = known_test_mask.numpy()
    test_mask[~test_mask] = known_test_mask
    # effective split:
    test_data = data[test_mask].copy()
    train_data = data[~test_mask].copy()

    assert test_data.shape[0] + train_data.shape[0] == data.shape[0]
    
    return train_data, test_data