import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm, trange
from sklearn.preprocessing import LabelEncoder
import os


class SyntheticBalancedDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            balancing_label: str):

        self.ZdA_mask = df.ZdA
        self.type_A_ZdA_mask = torch.from_numpy(
            df['Type_A_ZdA'].values).float().unsqueeze(-1)
        self.type_B_ZdA_mask = torch.from_numpy(
            df['Type_B_ZdA'].values).float().unsqueeze(-1)
        data_points = torch.from_numpy(
            df[['X', 'Y']].values)
        self.macro_labels = torch.from_numpy(
            df['Macro Label'].values).unsqueeze(-1)
        self.micro_labels = torch.from_numpy(
            df['Micro Label'].values).unsqueeze(-1)
        self.balancing_label = balancing_label

        if self.balancing_label == 'Micro Label':
            self.balancing_classes = self.micro_labels.unique()
        elif self.balancing_label == 'Macro Label':
            self.balancing_classes = self.macro_labels.unique()
        # build special label tensor where a label is a tensor
        # indicating the macro, micro class, and if it is a
        # type A or Type B class
        self.labels = torch.cat([
                   self.macro_labels,
                   self.micro_labels,
                   self.type_A_ZdA_mask,
                   self.type_B_ZdA_mask],
            dim=1)

        # for balanced sampling purposes.
        self.cache = self.get_cache(data_points)
        lens = [len(cs_slice[0]) for cs_slice in self.cache]
        self.length = np.array(lens).max() * len(self.balancing_classes)

    def get_cache(self, data_points):
        cache = []
        for class_idx in self.balancing_classes:
            if self.balancing_label == 'Micro Label':
                class_mask = self.micro_labels == class_idx
            if self.balancing_label == 'Macro Label':
                class_mask = self.macro_labels == class_idx
            cache.append((data_points[class_mask.squeeze(-1)],
                          self.labels[class_mask.squeeze(-1)]))
        return cache

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        slice_to_query = idx % len(self.balancing_classes)
        curr_slice = self.cache[slice_to_query]
        real_idx = int(idx / len(self.balancing_classes))
        slice_dim = curr_slice[0].shape[0]
        if idx != 0:
            real_idx = real_idx % slice_dim
        else:
            real_idx = 0
        return curr_slice[0][real_idx], curr_slice[1][real_idx]


class SyntheticBalancedDataset2(Dataset):
    def __init__(
            self,
            features,
            df: pd.DataFrame,
            balancing_label: str):

        self.ZdA_mask = df.ZdA
        self.type_A_ZdA_mask = torch.from_numpy(
            df['Type_A_ZdA'].values).float().unsqueeze(-1)
        self.type_B_ZdA_mask = torch.from_numpy(
            df['Type_B_ZdA'].values).float().unsqueeze(-1)
        data_points = torch.from_numpy(features)
        self.macro_labels = torch.from_numpy(
            df['Macro Label'].values).unsqueeze(-1)
        self.micro_labels = torch.from_numpy(
            df['Micro Label'].values).unsqueeze(-1)
        self.balancing_label = balancing_label

        if self.balancing_label == 'Micro Label':
            self.balancing_classes = self.micro_labels.unique()
        elif self.balancing_label == 'Macro Label':
            self.balancing_classes = self.macro_labels.unique()
        # build special label tensor where a label is a tensor
        # indicating the macro, micro class, and if it is a
        # type A or Type B class
        self.labels = torch.cat([
                   self.macro_labels,
                   self.micro_labels,
                   self.type_A_ZdA_mask,
                   self.type_B_ZdA_mask],
            dim=1)

        # for balanced sampling purposes.
        self.cache = self.get_cache(data_points)
        lens = [len(cs_slice[0]) for cs_slice in self.cache]
        self.length = np.array(lens).max() * len(self.balancing_classes)
        self.items = self.init_items()

    def init_items(self):
        items = []
        for idx in trange(self.length):
            items.append(self.getitem_balanced(idx))
        return items

    def get_cache(self, data_points):
        cache = []
        for class_idx in self.balancing_classes:
            if self.balancing_label == 'Micro Label':
                class_mask = self.micro_labels == class_idx
            if self.balancing_label == 'Macro Label':
                class_mask = self.macro_labels == class_idx
            cache.append((data_points[class_mask.squeeze(-1)],
                          self.labels[class_mask.squeeze(-1)]))
        return cache

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.items[idx]

    def getitem_balanced(self, idx):
        slice_to_query = idx % len(self.balancing_classes)
        curr_slice = self.cache[slice_to_query]
        real_idx = int(idx / len(self.balancing_classes))
        slice_dim = curr_slice[0].shape[0]
        if idx != 0:
            real_idx = real_idx % slice_dim
        else:
            real_idx = 0
        return curr_slice[0][real_idx], curr_slice[1][real_idx]


class RealBalancedDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            balancing_label: str,
            root_dir: str):

        self.root_dir = root_dir
        self.ZdA_mask = df.ZdA
        self.type_A_ZdA_mask = torch.from_numpy(
            df['Type_A_ZdA'].values).float().unsqueeze(-1)
        self.type_B_ZdA_mask = torch.from_numpy(
            df['Type_B_ZdA'].values).float().unsqueeze(-1)

        file_names = df['filename'].values

        self.macro_label_encoder = LabelEncoder()
        macro_encoded_labels = self.macro_label_encoder.fit_transform(
            df['Macro Label'].values)
        self.macro_encoded_labels = torch.Tensor([
            int(label) for label in macro_encoded_labels]).unsqueeze(-1)

        self.micro_label_encoder = LabelEncoder()
        micro_encoded_labels = self.micro_label_encoder.fit_transform(
            df['Micro Label'].values)
        self.micro_encoded_labels = torch.Tensor([
            int(label) for label in micro_encoded_labels]).unsqueeze(-1)

        self.balancing_label = balancing_label

        if self.balancing_label == 'Micro Label':
            self.balancing_classes = self.micro_encoded_labels.unique()
        elif self.balancing_label == 'Macro Label':
            self.balancing_classes = self.macro_encoded_labels.unique()
        # build special label tensor where a label is a tensor
        # indicating the macro, micro class, and if it is a
        # type A or Type B class
        self.labels = torch.cat([
                   self.macro_encoded_labels,
                   self.micro_encoded_labels,
                   self.type_A_ZdA_mask,
                   self.type_B_ZdA_mask],
            dim=1)

        # for balanced sampling purposes.
        self.cache = self.get_cache(file_names)
        lens = [len(cs_slice[0]) for cs_slice in self.cache]
        self.length = np.array(lens).max() * len(self.balancing_classes)
        self.items = self.init_items()

    def init_items(self):
        items = []
        for idx in trange(self.length):
            items.append(self.getitem_balanced(idx))
        return items

    def get_cache(self, file_names):
        cache = []
        for class_idx in self.balancing_classes:
            if self.balancing_label == 'Micro Label':
                class_mask = self.micro_encoded_labels == class_idx
            if self.balancing_label == 'Macro Label':
                class_mask = self.macro_encoded_labels == class_idx
            cache.append((file_names[class_mask.squeeze(-1)],
                          self.labels[class_mask.squeeze(-1)]))
        return cache

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        file_name, labels = self.items[idx]
        file_path = os.path.join(self.root_dir, file_name)
        image = torch.load(file_path)
        image = image.unsqueeze(0) / 255
        image = torch.cat([image, torch.zeros((2, 512, 512))])
        return image, labels

    def getitem_balanced(self, idx):
        slice_to_query = idx % len(self.balancing_classes)
        curr_slice = self.cache[slice_to_query]
        real_idx = int(idx / len(self.balancing_classes))
        slice_dim = curr_slice[0].shape[0]
        if idx != 0:
            real_idx = real_idx % slice_dim
        else:
            real_idx = 0

        file_name = curr_slice[0][real_idx]
        return file_name, curr_slice[1][real_idx]


class SynthFewShotDataset(Dataset):

    def __init__(
            self,
            features,
            df: pd.DataFrame):

        self.data_points = torch.from_numpy(features)
        data_idxs = np.arange(0, self.data_points.shape[0], dtype=np.int32)

        self.type_A_ZdA_mask = torch.from_numpy(
            df['Type_A_ZdA'].values).float().unsqueeze(-1)
        self.type_B_ZdA_mask = torch.from_numpy(
            df['Type_B_ZdA'].values).float().unsqueeze(-1)
        self.macro_labels = torch.from_numpy(
            df['Macro Label'].values).unsqueeze(-1)
        self.micro_labels = torch.from_numpy(
            df['Micro Label'].values).unsqueeze(-1)
        
        # we need to know which classes are known and unknown:
        self.type_A_micro_attacks = df[df['Type_A_ZdA']]['Micro Label'].unique()
        self.type_B_micro_attacks = df[df['Type_B_ZdA']]['Micro Label'].unique()
        self.type_A_macro_attacks = df[df['Type_A_ZdA']]['Macro Label'].unique()
        self.known_micro_attacks = df[df['ZdA'] == False]['Micro Label'].unique()
        self.known_macro_attacks = df[df['ZdA'] == False]['Macro Label'].unique()

        self.known_taxonomy = {}
        for k_macro in self.known_macro_attacks:
            for k_micro in df[(df['Macro Label']==k_macro) & (df['ZdA'] == False)]['Micro Label'].unique():
                if k_macro not in self.known_taxonomy.keys():
                    self.known_taxonomy[k_macro] = []
                self.known_taxonomy[k_macro].append(k_micro)


        self.type_B_taxonomy = {}
        for k_macro in self.known_macro_attacks:
            for k_micro in df[(df['Macro Label']==k_macro) & (df['ZdA'] == True)]['Micro Label'].unique():
                if k_macro not in self.type_B_taxonomy.keys():
                    self.type_B_taxonomy[k_macro] = []
                self.type_B_taxonomy[k_macro].append(k_micro)

        # build special label tensor where a label is a tensor
        # indicating the macro, micro class, and if it is a
        # type A or Type B class
        self.labels = torch.cat([
                   self.macro_labels,
                   self.micro_labels,
                   self.type_A_ZdA_mask,
                   self.type_B_ZdA_mask],
            dim=1)

        self.micro_classes = df['Micro Label'].unique()
        self.macro_classes = df['Macro Label'].unique()

        self.idxs_per_micro_class = {}
        for class_name in self.micro_classes:
            class_mask = self.micro_labels.squeeze(-1) == class_name
            idxs_of_class = data_idxs[class_mask]
            self.idxs_per_micro_class[class_name] = idxs_of_class


    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        return self.data_points[idx], self.labels[idx]

class Simple_Dataset(Dataset):

    def __init__(
            self,
            features,
            df: pd.DataFrame):

        self.data_points = torch.from_numpy(features)
        data_idxs = np.arange(0, self.data_points.shape[0], dtype=np.int32)

        self.ZdA_mask = torch.from_numpy(
            df['ZdA'].values).float().unsqueeze(-1)

        self.label_encoder = LabelEncoder()

        encoded_labels = self.label_encoder.fit_transform(
           df['Label'].values)

        self.encoded_labels = torch.Tensor([
            int(label) for label in encoded_labels]).long().unsqueeze(-1)
        
        # we need to know which classes are known and unknown:
        self.zdas = df[df['ZdA']]['Label'].unique()
        self.known_attacks = df[df['ZdA'] == False]['Label'].unique()

        self.zdas = self.label_encoder.transform(self.zdas).astype(int)
        self.known_attacks = self.label_encoder.transform(self.known_attacks).astype(int)


        # build special label tensor where a label is a tensor
        # indicating the class, and if it is a
        # type A or Type B class
        self.labels = torch.cat([
                   self.encoded_labels,
                   self.ZdA_mask],
            dim=1)

        self.classes = np.unique(self.encoded_labels)

        self.idxs_per_class = {}
        for class_name in self.classes:
            class_mask = self.labels[:,0] == class_name
            idxs_of_class = data_idxs[class_mask]
            self.idxs_per_class[class_name] = idxs_of_class


    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        return self.data_points[idx], self.labels[idx]


class RealFewShotDataset_LowDim(Dataset):

    def __init__(
            self,
            features,
            df: pd.DataFrame,
            micro_label_enc,
            macro_label_enc):

        self.data_points = torch.from_numpy(features)
        data_idxs = np.arange(0, self.data_points.shape[0], dtype=np.int32)

        self.micro_labels = df['Micro Label'].values
        self.macro_labels = df['Macro Label'].values

        self.macro_label_encoder = macro_label_enc
        macro_encoded_labels = self.macro_label_encoder.transform(
           self.macro_labels)
        self.macro_encoded_labels = torch.Tensor([
            int(label) for label in macro_encoded_labels]).unsqueeze(-1)

        self.micro_label_encoder = micro_label_enc
        micro_encoded_labels = self.micro_label_encoder.transform(
            self.micro_labels)
        self.micro_encoded_labels = torch.Tensor([
            int(label) for label in micro_encoded_labels]).unsqueeze(-1)


        self.type_A_ZdA_mask = torch.from_numpy(
            df['Type_A_ZdA'].values).float().unsqueeze(-1)
        self.type_B_ZdA_mask = torch.from_numpy(
            df['Type_B_ZdA'].values).float().unsqueeze(-1)

        # we need to know which classes are known and unknown:
        self.type_A_micro_attacks = df[df['Type_A_ZdA']]['Micro Label'].unique()
        self.type_B_micro_attacks = df[df['Type_B_ZdA']]['Micro Label'].unique()
        self.type_A_macro_attacks = df[df['Type_A_ZdA']]['Macro Label'].unique()
        self.known_micro_attacks = df[df['ZdA'] == False]['Micro Label'].unique()
        self.known_macro_attacks = df[df['ZdA'] == False]['Macro Label'].unique()


        self.known_taxonomy = {}
        for k_macro in self.known_macro_attacks:
            for k_micro in df[(df['Macro Label']==k_macro) & (df['ZdA'] == False)]['Micro Label'].unique():
                if k_macro not in self.known_taxonomy.keys():
                    self.known_taxonomy[k_macro] = []
                self.known_taxonomy[k_macro].append(k_micro)


        self.type_B_taxonomy = {}
        for k_macro in self.known_macro_attacks:
            for k_micro in df[(df['Macro Label']==k_macro) & (df['ZdA'] == True)]['Micro Label'].unique():
                if k_macro not in self.type_B_taxonomy.keys():
                    self.type_B_taxonomy[k_macro] = []
                self.type_B_taxonomy[k_macro].append(k_micro)

        # build special label tensor where a label is a tensor
        # indicating the macro, micro class, and if it is a
        # type A or Type B class
        self.labels = torch.cat([
                   self.macro_encoded_labels,
                   self.micro_encoded_labels,
                   self.type_A_ZdA_mask,
                   self.type_B_ZdA_mask],
            dim=1)

        self.micro_classes = np.unique(self.micro_labels)

        self.idxs_per_micro_class = {}
        for class_name in self.micro_classes:
            class_mask = self.micro_labels == class_name
            idxs_of_class = data_idxs[class_mask]
            self.idxs_per_micro_class[class_name] = idxs_of_class


    def __len__(self):
        return len(self.data_points)


    def __getitem__(self, idx):
        return self.data_points[idx], self.labels[idx]


class RealFewShotDataset(Dataset):

    def __init__(
            self,
            root_dir: str,
            df: pd.DataFrame):

        self.root_dir = root_dir
        self.file_names = df['filename'].values
        data_idxs = np.arange(0, len(self.file_names), dtype=np.int32)

        self.micro_labels = df['Micro Label'].values
        self.macro_labels = df['Macro Label'].values

        self.macro_label_encoder = LabelEncoder()
        macro_encoded_labels = self.macro_label_encoder.fit_transform(
           self.macro_labels)
        self.macro_encoded_labels = torch.Tensor([
            int(label) for label in macro_encoded_labels]).unsqueeze(-1)

        self.micro_label_encoder = LabelEncoder()
        micro_encoded_labels = self.micro_label_encoder.fit_transform(
            self.micro_labels)
        self.micro_encoded_labels = torch.Tensor([
            int(label) for label in micro_encoded_labels]).unsqueeze(-1)


        self.type_A_ZdA_mask = torch.from_numpy(
            df['Type_A_ZdA'].values).float().unsqueeze(-1)
        self.type_B_ZdA_mask = torch.from_numpy(
            df['Type_B_ZdA'].values).float().unsqueeze(-1)

        # we need to know which classes are known and unknown:
        self.type_A_micro_attacks = df[df['Type_A_ZdA']]['Micro Label'].unique()
        self.type_B_micro_attacks = df[df['Type_B_ZdA']]['Micro Label'].unique()
        self.type_A_macro_attacks = df[df['Type_A_ZdA']]['Macro Label'].unique()
        self.known_micro_attacks = df[df['ZdA'] == False]['Micro Label'].unique()
        self.known_macro_attacks = df[df['ZdA'] == False]['Macro Label'].unique()


        self.known_taxonomy = {}
        for k_macro in self.known_macro_attacks:
            for k_micro in df[(df['Macro Label']==k_macro) & (df['ZdA'] == False)]['Micro Label'].unique():
                if k_macro not in self.known_taxonomy.keys():
                    self.known_taxonomy[k_macro] = []
                self.known_taxonomy[k_macro].append(k_micro)


        self.type_B_taxonomy = {}
        for k_macro in self.known_macro_attacks:
            for k_micro in df[(df['Macro Label']==k_macro) & (df['ZdA'] == True)]['Micro Label'].unique():
                if k_macro not in self.type_B_taxonomy.keys():
                    self.type_B_taxonomy[k_macro] = []
                self.type_B_taxonomy[k_macro].append(k_micro)

        # build special label tensor where a label is a tensor
        # indicating the macro, micro class, and if it is a
        # type A or Type B class
        self.labels = torch.cat([
                   self.macro_encoded_labels,
                   self.micro_encoded_labels,
                   self.type_A_ZdA_mask,
                   self.type_B_ZdA_mask],
            dim=1)

        self.micro_classes = np.unique(self.micro_labels)

        self.idxs_per_micro_class = {}
        for class_name in self.micro_classes:
            class_mask = self.micro_labels == class_name
            idxs_of_class = data_idxs[class_mask]
            self.idxs_per_micro_class[class_name] = idxs_of_class


    def __len__(self):
        return len(self.file_names)


    def __getitem__(self, idx):
        
        labels = self.labels[idx]
        if not isinstance(idx, int):
            items = []
            for real_idx in idx:
                items.append(self.load_single_image(real_idx))
            image = torch.cat(items, 0)
        else:
            image = self.load_single_image(idx)
        return image, labels

    def load_single_image(self, idx):
        file_name = self.file_names[idx]
        file_path = os.path.join(self.root_dir, file_name)
        image = torch.load(file_path)
        image = image.unsqueeze(0) / 255
        image = torch.cat([image, torch.zeros((2, 512, 512))]).unsqueeze(0)
        return image

        
def convenient_cf(batch):
    """
    No matter what you do, you can't do something like this
    in the sampler. 
    """
    return batch[0] 
    
    
class FewShotSampler(Sampler):
    def __init__(
            self,
            dataset,
            n_tasks,
            classes_per_it,
            k_shot,
            q_shot):

        self.n_tasks = n_tasks
        self.dataset = dataset
        
        # At each batch, we insert 1 Type A Zda and 1 Type B ZdB.
        self.known_classes_per_it = classes_per_it - 2

        self.k_shot = k_shot
        self.q_shot = q_shot

    def reset(self):
        np.random.seed(42)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):

        for _ in range(self.n_tasks):
            support_idxs = np.array([], dtype=np.int32)
            query_idxs = np.array([], dtype=np.int32)

            """
            We need to choose self.known_classes_per_it
            known classes from self.known_classes_per_it diverse
            macro clusters, because we are doing fixed-way few-shot learning
            both in the micro and macro realm.
            """
            choosen_macros = np.random.choice(
               self.dataset.known_macro_attacks,
               self.known_classes_per_it,
               replace=False 
            )

            choosen_known_micros = []
            for macro in choosen_macros:
                choosen_known_micros.append(
                        np.random.choice(
                            self.dataset.known_taxonomy[macro],
                            1)[0])
            
            for known_micro in choosen_known_micros:
                class_idxs = np.random.choice(
                    self.dataset.idxs_per_micro_class[known_micro],
                    self.k_shot + self.q_shot,
                    replace=False)

                support_idxs = np.concatenate(
                    (support_idxs, class_idxs[:self.k_shot]))
                query_idxs = np.concatenate(
                    (query_idxs, class_idxs[self.k_shot:]))

            """
            Type B micro attacks for Few Shot batch should not be 
            from any of the previous choosen micro classes because 
            otherwise we will have a macro-level collision in phase 2.
            """
            remaining_macros = list(
                set(self.dataset.type_B_taxonomy.keys()) - set(choosen_macros))
            curr_type_B_macro = np.random.choice(
                remaining_macros,1)[0]

            # Type B class:
            type_B_class_name = np.random.choice(
                self.dataset.type_B_taxonomy[curr_type_B_macro],
                1)[0]

            class_idxs = np.random.choice(
                    self.dataset.idxs_per_micro_class[type_B_class_name],
                    self.k_shot + self.q_shot)

            support_idxs = np.concatenate(
                    (support_idxs, class_idxs[:self.k_shot]))
            query_idxs = np.concatenate(
                    (query_idxs, class_idxs[self.k_shot:]))    

            # Type A class:
            type_A_class_name = np.random.choice(
                self.dataset.type_A_micro_attacks,
                1)[0]

            class_idxs = np.random.choice(
                    self.dataset.idxs_per_micro_class[type_A_class_name],
                    self.k_shot + self.q_shot)

            support_idxs = np.concatenate(
                (support_idxs, class_idxs[:self.k_shot]))
            query_idxs = np.concatenate(
                (query_idxs, class_idxs[self.k_shot:]))

            yield np.concatenate((support_idxs, query_idxs))


class FewShotSamplerReal(Sampler):
    def __init__(
            self,
            dataset,
            n_tasks,
            classes_per_it,
            k_shot,
            q_shot):

        self.n_tasks = n_tasks
        self.dataset = dataset
        
        # At each batch, we insert 1 Type A Zda and 1 Type B ZdB.
        self.known_classes_per_it = classes_per_it - 2

        self.k_shot = k_shot
        self.q_shot = q_shot

    def reset(self):
        np.random.seed(42)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):

        for _ in range(self.n_tasks):
            support_idxs = np.array([], dtype=np.int32)
            query_idxs = np.array([], dtype=np.int32)

            """
            We need to choose self.known_classes_per_it
            known classes from self.known_classes_per_it diverse
            macro clusters, because we are doing fixed-way few-shot learning
            both in the micro and macro realm.
            """
            choosen_macros = np.random.choice(
               self.dataset.known_macro_attacks,
               self.known_classes_per_it,
               replace=False 
            )

            choosen_known_micros = []
            for macro in choosen_macros:
                choosen_known_micros.append(
                        np.random.choice(
                            self.dataset.known_taxonomy[macro],
                            1)[0])
            
            for known_micro in choosen_known_micros:
                class_idxs = np.random.choice(
                    self.dataset.idxs_per_micro_class[known_micro],
                    self.k_shot + self.q_shot,
                    replace=False)

                support_idxs = np.concatenate(
                    (support_idxs, class_idxs[:self.k_shot]))
                query_idxs = np.concatenate(
                    (query_idxs, class_idxs[self.k_shot:]))

            """
            Type B micro attacks for Few Shot batch should not be 
            from any of the previous choosen micro classes because 
            otherwise we will have a macro-level collision in phase 2.
            """
            remaining_macros = list(
                set(self.dataset.type_B_taxonomy.keys()) - set(choosen_macros))
            curr_type_B_macro = np.random.choice(
                remaining_macros,1)[0]

            # Type B class:
            type_B_class_name = np.random.choice(
                self.dataset.type_B_taxonomy[curr_type_B_macro],
                1)[0]

            class_idxs = np.random.choice(
                    self.dataset.idxs_per_micro_class[type_B_class_name],
                    self.k_shot + self.q_shot)

            support_idxs = np.concatenate(
                    (support_idxs, class_idxs[:self.k_shot]))
            query_idxs = np.concatenate(
                    (query_idxs, class_idxs[self.k_shot:]))    

            # Type A class:
            type_A_class_name = np.random.choice(
                self.dataset.type_A_micro_attacks,
                1)[0]

            class_idxs = np.random.choice(
                    self.dataset.idxs_per_micro_class[type_A_class_name],
                    self.k_shot + self.q_shot)

            support_idxs = np.concatenate(
                (support_idxs, class_idxs[:self.k_shot]))
            query_idxs = np.concatenate(
                (query_idxs, class_idxs[self.k_shot:]))

            """
            # DEBUG

            print(f'choosen macros: {choosen_macros}')
            print(f'choosen_known_micros: {choosen_known_micros}')
            print(f'remaining_macros: {remaining_macros}')
            print(f'curr_type_B_macro: {curr_type_B_macro}')
            print(f'type_B_class_name: {type_B_class_name}')
            print(f'type_A_class_name: {type_A_class_name}')
            """

            yield np.concatenate((support_idxs, query_idxs))


class FewShotSampler_Simple(Sampler):
    def __init__(
            self,
            dataset,
            n_tasks,
            classes_per_it,
            k_shot,
            q_shot):

        self.n_tasks = n_tasks
        self.dataset = dataset
        
        # At each batch, we insert 1 Zda
        self.known_classes_per_it = classes_per_it - 1

        self.k_shot = k_shot
        self.q_shot = q_shot

    def reset(self):
        np.random.seed(42)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):

        for _ in range(self.n_tasks):
            support_idxs = np.array([], dtype=np.int32)
            query_idxs = np.array([], dtype=np.int32)


            choosen_classes = np.random.choice(
               self.dataset.known_attacks,
               self.known_classes_per_it,
               replace=False 
            )
            
            for known_attacks in choosen_classes:
                class_idxs = np.random.choice(
                    self.dataset.idxs_per_class[known_attacks],
                    self.k_shot + self.q_shot,
                    replace=False)

                support_idxs = np.concatenate(
                    (support_idxs, class_idxs[:self.k_shot]))
                query_idxs = np.concatenate(
                    (query_idxs, class_idxs[self.k_shot:]))


            curr_zda = np.random.choice(
                self.dataset.zdas,1)[0]

            class_idxs = np.random.choice(
                    self.dataset.idxs_per_class[curr_zda],
                    self.k_shot + self.q_shot)

            support_idxs = np.concatenate(
                    (support_idxs, class_idxs[:self.k_shot]))
            query_idxs = np.concatenate(
                    (query_idxs, class_idxs[self.k_shot:]))    

            yield np.concatenate((support_idxs, query_idxs))