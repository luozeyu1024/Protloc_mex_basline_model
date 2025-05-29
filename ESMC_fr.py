
from esm.models.esmc import ESMC  
import torch
import torch.nn as nn

from esm.pretrained import LOCAL_MODEL_REGISTRY  # Import registry
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
from esm.sdk.api import ESMProtein, LogitsConfig


class EsmcLastHiddenFeatureExtractor:
    def __init__(self,  compute_cls=True, compute_eos=True, compute_mean=True,
                 compute_segments=False, num_segments=10, device_choose='auto'):
        self.compute_cls = compute_cls
        self.compute_eos = compute_eos
        self.compute_mean = compute_mean
        self.compute_segments = compute_segments
        self.num_segments = num_segments

        if device_choose == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device_choose.startswith('cuda'):
            if torch.cuda.is_available():
                device_id = int(device_choose.split(':')[-1]) if ':' in device_choose else 0
                self.device = torch.device(f"cuda:{device_id}")
            else:
                raise TypeError("CUDA is not available.")
        elif device_choose == 'cpu':
            self.device = torch.device("cpu")
        else:
            raise ValueError("Invalid device choice. Use 'auto', 'cpu', or 'cuda[:<id>]'.")

    def get_last_cls_token(self, embeddings):
        return embeddings[:, 0, :]

    def get_last_eos_token(self, embeddings, eos_position):
        return embeddings[:, eos_position, :]

    def get_last_mean_token(self, embeddings, eos_position):
        embeddings = embeddings
        return embeddings[:, 1:eos_position, :].mean(dim=1)

    def get_segment_mean_tokens(self, embeddings, eos_position):
        embeddings = embeddings
        seq_len = eos_position - 1
        segment_size, remainder = divmod(seq_len, self.num_segments)
        segment_means = []

        start = 1
        for i in range(self.num_segments):
            end = start + segment_size + (1 if i < remainder else 0)

            if end > start:
                segment_mean = embeddings[:, start:end, :].mean(dim=1)
            else:
                segment_mean = torch.zeros(embeddings[:, start:start + 1, :].shape,
                                           device=embeddings.device)

            segment_means.append(segment_mean.squeeze().tolist())
            start = end

        return segment_means

    def get_last_hidden_features_combine(self, X_input, sequence_name='sequence'):
        X_input = X_input.reset_index(drop=True)

        sequences = X_input[sequence_name].tolist()

        features_length = {}
        columns = None
        all_results = []
        # Call from_pretrained with registered model name
        client = ESMC.from_pretrained("custom-esmc-600m",device=self.device)

        with torch.no_grad():
            for idx, seq in tqdm(enumerate(sequences), desc='Sequences inference', total=len(sequences)):

                # Use ESMC model API encode method
                protein = ESMProtein(sequence=seq)


                # Encode protein to get tensor representation
                protein_tensor = client.encode(protein)

                logits_output = client.logits(
                    protein_tensor,
                    LogitsConfig(
                        sequence=True,
                        return_embeddings=True,
                        return_hidden_states=False
                    )
                )

                embeddings = logits_output.embeddings  # Use embeddings to extract features without layerNorm normalization

                eos_position = len(seq) + 1

                features = []
                if self.compute_cls:
                    cls_features = self.get_last_cls_token(embeddings).squeeze().tolist()
                    features_length.setdefault('cls', len(cls_features))
                    features.extend(cls_features)

                if self.compute_eos:
                    eos_features = self.get_last_eos_token(embeddings, eos_position).squeeze().tolist()
                    features_length.setdefault('eos', len(eos_features))
                    features.extend(eos_features)

                if self.compute_mean:
                    mean_features = self.get_last_mean_token(embeddings, eos_position).squeeze().tolist()
                    features_length.setdefault('mean', len(mean_features))
                    features.extend(mean_features)

                if self.compute_segments:
                    segment_means = self.get_segment_mean_tokens(embeddings, eos_position)
                    for seg, segment_mean in enumerate(segment_means):
                        features.extend(segment_mean)
                        features_length.setdefault(f'segment{seg}_mean', len(segment_mean))

                if columns is None:
                    columns = [f"ESMC_{ftype}{k}" for ftype, length in features_length.items() for k in range(length)]

                result = pd.DataFrame([features], columns=columns, index=[idx])
                all_results.append(result)

                del protein_tensor, logits_output
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

        X_outcome = pd.concat(all_results, axis=0)

        print(f'Features dimensions: {features_length}')

        combined_result = pd.concat([X_input, X_outcome], axis=1)
        return combined_result




    def get_all_token_embeddings(self, X_input, sequence_name='sequence', id_column=None):
        """
        For each protein sequence, compute the embedding of all tokens and output a dictionary.
        The key is the protein ID (if not provided, use the row index), and the value is the corresponding numpy array (shape: [number of tokens, embedding dimension]).

        Parameters:
            X_input: pandas DataFrame containing protein sequences.
            sequence_name: column name storing protein sequences, default is 'sequence'.
            id_column: column name for protein ID. If provided, use this column as the key; otherwise, use the DataFrame row index.

        Returns:
            embeddings_dict: dict, {protein_id: numpy.ndarray}
        """
        embeddings_dict = {}
        X_input = X_input.reset_index(drop=True)

        with torch.no_grad():

            # Use pretrained model (ESMC), model name is "custom-esmc-600m"
            client = ESMC.from_pretrained("custom-esmc-600m").to(self.device)

            # Iterate over each protein record in the DataFrame
            for idx, row in tqdm(X_input.iterrows(), total=len(X_input), desc='Protein embeddings extraction'):
                seq = row[sequence_name]
                # Use provided protein ID, if not provided use row index
                protein_id = row[id_column] if id_column is not None else idx
                # Construct protein object (assuming ESMProtein is defined)
                protein = ESMProtein(sequence=seq)

                # Encode protein to get tensor representation
                protein_tensor = client.encode(protein)
                # Call logits method with config: sequence=True, return_embeddings=True, return_hidden_states=False
                logits_output = client.logits(
                    protein_tensor,
                    LogitsConfig(
                        sequence=True,
                        return_embeddings=True,
                        return_hidden_states=False
                    )
                )
                # Get all token embeddings
                # logits_output.embeddings shape is (1, seq_length, embed_dim)
                embeddings = logits_output.embeddings
                # Remove batch dimension, get (seq_length, embed_dim)
                embeddings_np = embeddings[0].cpu().numpy()
                # Store in dictionary, key is protein ID, value is corresponding numpy array
                embeddings_dict[protein_id] = embeddings_np

                del protein_tensor, logits_output
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
        return embeddings_dict


    def get_combined_features_and_all_token_embeddings(self, X_input, sequence_name='sequence', id_column=None):
        """
        Simultaneously extract features (CLS, EOS, Mean, segment Mean) and all token embeddings.

        Parameters:
            X_input: pandas DataFrame containing protein sequences.
            sequence_name: column name storing protein sequences, default is 'sequence'.
            id_column: column name for protein ID; if provided, use this column as the key, otherwise use the DataFrame row index.

        Returns:
            tuple: (combined_result, embeddings_dict)
                - combined_result: DataFrame with original data concatenated with extracted features.
                - embeddings_dict: Dictionary containing all token embeddings, key is protein ID, value is the corresponding numpy array.
        """
        # Reset index to ensure subsequent row numbers are consistent
        X_input = X_input.reset_index(drop=True)
        features_list = []  # Store feature vectors for each protein
        feature_names = None  # Feature name list to be determined
        embeddings_dict = {}  # Store all token embeddings for each protein


        # Use registered model (note to pass device parameter)
        client = ESMC.from_pretrained("custom-esmc-600m", device=self.device)

        # Iterate over each protein record
        for index, row in tqdm(X_input.iterrows(), total=len(X_input), desc='Processing proteins'):
            seq = row[sequence_name]
            # If id_column is provided, use its value; otherwise, use current row index
            protein_id = row[id_column] if id_column is not None else index

            # Construct protein object and encode to get tensor representation
            protein = ESMProtein(sequence=seq)
            protein_tensor = client.encode(protein)
            logits_output = client.logits(
                protein_tensor,
                LogitsConfig(
                    sequence=True,
                    return_embeddings=True,
                    return_hidden_states=False
                )
            )
            embeddings = logits_output.embeddings  # shape: (1, seq_length, embed_dim)
            # Save full token embedding (remove batch dimension)
            embeddings_np = embeddings[0].cpu().numpy()
            embeddings_dict[protein_id] = embeddings_np

            # Determine EOS token position based on input sequence length:
            # For BERT-like models, [CLS] is at index 0, followed by sequence tokens,
            # If sequence length is n, EOS is at index n+1, so: eos_position = len(seq) + 1
            eos_position = len(seq) + 1

            # --- Feature extraction section ---
            # For each protein, extract required features in order, each feature is a d-dimensional vector,
            # Let d = embeddings.shape[-1]
            d = embeddings.shape[-1]
            current_features = []

            if self.compute_cls:
                # CLS feature: index 0
                cls_feature = self.get_last_cls_token(embeddings).squeeze().tolist()
                current_features.extend(cls_feature)

            if self.compute_eos:
                # EOS feature: index eos_position
                eos_feature = self.get_last_eos_token(embeddings, eos_position).squeeze().tolist()
                current_features.extend(eos_feature)

            if self.compute_mean:
                # Mean feature: mean of tokens [1, eos_position)
                mean_feature = self.get_last_mean_token(embeddings, eos_position).squeeze().tolist()
                current_features.extend(mean_feature)

            if self.compute_segments:
                # Segment mean features: divide token sequence into num_segments segments,
                # Compute mean for each segment, formula:
                #   segment_size = floor((n) / num_segments)
                #   remainder = n mod num_segments
                # Each segment mean is a d-dimensional vector
                segment_features = self.get_segment_mean_tokens(embeddings, eos_position)
                for seg_feature in segment_features:
                    current_features.extend(seg_feature)

            # For the first sample, construct the full feature name list based on current embedding dimension
            if feature_names is None and current_features:
                names = []
                if self.compute_cls:
                    names.extend([f"ESMC_cls{k}" for k in range(d)])
                if self.compute_eos:
                    names.extend([f"ESMC_eos{k}" for k in range(d)])
                if self.compute_mean:
                    names.extend([f"ESMC_mean{k}" for k in range(d)])
                if self.compute_segments:
                    for seg in range(self.num_segments):
                        names.extend([f"ESMC_segment{seg}_mean{k}" for k in range(d)])
                feature_names = names

            features_list.append(current_features)

            # Clear GPU memory
            del protein_tensor, logits_output
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # Print extracted feature dimensions

        # Construct DataFrame from features_list and concatenate with original data
        if feature_names is not None and len(features_list) > 0:
            features_df = pd.DataFrame(features_list, columns=feature_names)
            combined_result = pd.concat([X_input, features_df], axis=1)

        else:
            combined_result = X_input.copy()
            print('No features extracted.')

        return combined_result, embeddings_dict

    def extract_residue_features(self, df, embeddings_dict, id_column, position_column, sequence_column=None, mode='strict'):
        """
        Extract features of specific amino acid residue positions from precomputed protein embeddings.
        
        Parameters:
            df: DataFrame containing UniportID, protein sequence, and residue position.
            embeddings_dict: Dictionary containing protein embeddings, key is protein ID, value is embedding tensor.
            id_column: Name of the UniportID column.
            sequence_column: Name of the protein sequence column (required in 'strict' mode; can be None in 'fast' mode).
            position_column: Name of the residue position column. The position to extract is counted from 1 (excluding CLS). Python index starts from 0.
            mode: 'strict' or 'fast'. In 'strict' mode, the embedding tensor length is strictly checked against the sequence length; in 'fast' mode, embeddings are extracted directly without sequence info.
            
        Returns:
            DataFrame containing the extracted features.
        """
        results = []
        
        # Group by protein ID for processing
        for protein_id, group in tqdm(df.groupby(id_column), desc="Processing Proteins"):
            if protein_id not in embeddings_dict:
                print(f"Warning: Protein ID {protein_id} not found in embedding dictionary, skipping")
                continue
                
            embeddings = embeddings_dict[protein_id]
            

            if mode == 'strict':
                if sequence_column is None:
                    raise ValueError("sequence_column must be provided in strict mode")
                # In strict mode, get the protein sequence and strictly check the embedding length; check if embedding tensor length equals sequence length + 2 (considering CLS and EOS tokens)
                protein_seq = group[sequence_column].iloc[0]

                expected_length = len(protein_seq) + 2
                if embeddings.shape[0] != expected_length:
                    print(f"Error: Embedding tensor length ({embeddings.shape[0]}) for protein ID {protein_id} does not match expected length ({expected_length}). Please download the reference protein sequence from UniProt.")
                    continue


            # Process all residue positions for this protein
            for _, row in group.iterrows():
                position = int(row[position_column])
                
                # Check if the position is valid (positions start from 1, but embedding index starts from 0, and there is a CLS token)
                if mode == 'strict':
                    if position < 1 or position > len(protein_seq):
                        print(f"Warning: Residue position {position} for protein ID {protein_id} is out of sequence range (1-{len(protein_seq)})")
                        continue
                
                # Extract feature (position is as given because index 0 is CLS token)
                try:
                    feature_vector = embeddings[position]
                    
                    # Create result row
                    result_row = row.copy()
                    # Add feature columns
                    for i, value in enumerate(feature_vector):
                        result_row[f'Residue_feature_{i}'] = value
                        
                    results.append(result_row)
                except IndexError:
                    if mode == 'strict':
                        # This should not happen in strict mode, as we have already checked the length
                        print(f"Error: Unable to extract feature at position {position} for protein ID {protein_id}")
                    else:
                        print(f"Warning: Residue position {position} for protein ID {protein_id} is out of embedding range (0-{embeddings.shape[0]-1})")
        
        if not results:
            print("Warning: No features were successfully extracted")
            return pd.DataFrame()
            
        return pd.DataFrame(results)



# Example: Feature extraction using example sequences ['AAAAA', 'AAA', 'AARA', 'AAAXA']
if __name__ == '__main__':

    # Define model file paths
    WEIGHTS_PATH = "D:/desktop/big_model/esmc-600m/data/weights/esmc_600m_2024_12_v0.pth"
    TOKENIZER_PATH = "D:/desktop/big_model/esm2_t6_8M_UR50D"


    def custom_ESMC_600M(device: torch.device | str = "cpu", use_flash_attn: bool = True) -> nn.Module:
        # Initialize model architecture
        with torch.device(device):
            # Load HuggingFace AutoTokenizer directly
            tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
            model = ESMC(
                d_model=1152,
                n_heads=18,
                n_layers=36,
                tokenizer=tokenizer,
                use_flash_attn=use_flash_attn,
            ).eval()
        # Load pretrained weights from specified path
        state_dict = torch.load(
            WEIGHTS_PATH,
            map_location=device,
        )
        model.load_state_dict(state_dict)
        return model


    # Register the custom loading function to LOCAL_MODEL_REGISTRY with the identifier "custom-esmc-600m"
    LOCAL_MODEL_REGISTRY["custom-esmc-600m"] = custom_ESMC_600M



    ##step1 Classic feature extraction: cls, eos, mean, global average pooling, segment features
    extractor = EsmcLastHiddenFeatureExtractor(
        compute_cls=True,
        compute_eos=True,
        compute_mean=True,
        compute_segments=True,  # If you need 1/10 segment average
        num_segments=10,device_choose='cpu')

    # Construct a DataFrame containing example sequences (column name 'sequence')
    data = pd.DataFrame({
        'sequence': ['AAAAAAAAAAARAA', 'AAA', 'AARA', 'AAAXA']
    })

    # Call the feature extraction function and merge original data with extracted features
    result = extractor.get_last_hidden_features_combine(data, sequence_name='sequence')

    # Output the result DataFrame
    print("Final merged result:")
    print(result)

    ##step2 Extract all token embeddings (including cls and eos) for each dictionary
    # Construct a simple DataFrame example containing protein sequences and corresponding IDs
    data = {
        "protein_id": ["P1", "P2"],
        "sequence": ["MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", "GILGFVFTLTVPSER"]
    }
    df = pd.DataFrame(data)

    extractor = EsmcLastHiddenFeatureExtractor(device_choose='cpu')
    embeddings_dict = extractor.get_all_token_embeddings(df, sequence_name='sequence', id_column='protein_id')

    for prot_id, emb in embeddings_dict.items():
        print(f"Protein ID: {prot_id}, Embedding shape: {emb.shape}")


    ##step3 Save and load embedding dictionary
    import numpy as np
    
    # Save embedding dictionary to npz file
    np.savez("embeddings_dict.npz", **embeddings_dict)
    print("Embedding dictionary saved to embeddings_dict.npz")
    
    # Load embedding dictionary
    loaded_data = np.load("embeddings_dict.npz")
    loaded_embeddings_dict = {key: loaded_data[key] for key in loaded_data}
    print("Loaded embedding dictionary, containing protein IDs:", list(loaded_embeddings_dict.keys()))
    
    ##step4 Simulate a DataFrame containing UniportID, protein sequence, and residue position
    # Create example data, one protein ID may have multiple positions to extract
    residue_data = {
        "UniportID": ["P1", "P1", "P1", "P2", "P2"],
        "sequence": ["MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", 
                    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", "GILGFVFTLTVPSER", "GILGFVFTLTVPSER"],
        "position": [5, 10, 15, 3, 8]  # Residue positions to extract features
    }
    residue_df = pd.DataFrame(residue_data)
    print("Example residue position DataFrame:")
    print(residue_df)
    
    # Extract features for specific residue positions using strict mode
    strict_features = extractor.extract_residue_features(
        residue_df, 
        loaded_embeddings_dict,
        id_column="UniportID",
        position_column="position",
        sequence_column="sequence",
        mode="strict"
    )
    print("\nFeatures extracted in strict mode (first 5 columns):")
    if not strict_features.empty:
        print(strict_features.iloc[:, :5])
    
    # Extract features for specific residue positions using fast mode
    fast_features = extractor.extract_residue_features(
        residue_df, 
        loaded_embeddings_dict,
        id_column="UniportID",
        position_column="position",
        mode="fast"
    )
    print("\nFeatures extracted in fast mode (first 5 columns):")
    if not fast_features.empty:
        print(fast_features.iloc[:, :5])

    # Output to Excel files
    strict_features.to_excel("strict_features.xlsx", index=False)
    fast_features.to_excel("fast_features.xlsx", index=False)
    print("\nFeature data has been saved to strict_features.xlsx and fast_features.xlsx respectively")







    # ===== Step5: Example usage of get_combined_features_and_all_token_embeddings =====

    # Construct a DataFrame example containing protein sequences and corresponding IDs
    data5 = pd.DataFrame({
        "protein_id": ['P1','P2','P3','P4'],
        'sequence': ['AAAAAAAAAAARAA', 'AAA', 'AARA', 'AAAXA']

    })

    # Create extractor instance, enable all feature extraction options, and use CPU device
    extractor = EsmcLastHiddenFeatureExtractor(
        compute_cls=True,
        compute_eos=True,
        compute_mean=True,
        compute_segments=True,
        num_segments=10,  # Set number of segments to 10
        device_choose='cpu'
    )

    # Call the function to get combined feature result DataFrame and all token embedding dictionary
    combined_result, embeddings_dict = extractor.get_combined_features_and_all_token_embeddings(
        X_input=data5,
        sequence_name='sequence',
        id_column='protein_id'
    )

    # Output the final combined result DataFrame
    print("Step5 - Combined Features DataFrame:")
    print(combined_result)

    # Output the shape of token embeddings for each protein
    print("\nStep5 - Token Embeddings Shapes:")
    for prot_id, emb in embeddings_dict.items():
        print(f"Protein ID: {prot_id}, Embedding Shape: {emb.shape}")
