#!/usr/bin/env python3
"""
LEO Satellite Attack Detection - Inference Script
Easy-to-use inference interface for trained ST-GNN model
"""

import torch
import numpy as np
import pickle
import sys

class SatelliteAttackDetector:
    """
    Satellite Attack Detection Model Interface
    
    Usage:
        detector = SatelliteAttackDetector('best_efficient_model.pt', 'processed_data.pkl')
        prediction, confidence = detector.predict(sequence_data)
    """
    
    def __init__(self, model_path, data_path):
        """
        Initialize detector with trained model and preprocessing objects
        
        Args:
            model_path: Path to trained model weights (.pt file)
            data_path: Path to processed data with scaler and label encoder (.pkl file)
        """
        # Import model architecture
        from efficient_stgnn_05 import EfficientSTGNN
        
        # Load preprocessing objects
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)
        
        self.scaler = data_dict['scaler']
        self.label_encoder = data_dict['label_encoder']
        self.feature_cols = data_dict['feature_cols']
        self.sequence_length = data_dict['sequence_length']
        
        # Initialize and load model
        self.model = EfficientSTGNN(
            num_features=len(self.feature_cols),
            num_classes=len(self.label_encoder.classes_),
            hidden_dim=64
        )
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Device: {self.device}")
        print(f"✓ Attack classes: {self.label_encoder.classes_}")
        print(f"✓ Required features: {len(self.feature_cols)}")
        print(f"✓ Sequence length: {self.sequence_length}")
    
    def predict(self, sequence, return_probabilities=False):
        """
        Predict attack type for a given sequence
        
        Args:
            sequence: numpy array of shape (sequence_length, num_features) or (num_features,)
                     If 1D, will be repeated to create sequence
            return_probabilities: If True, return all class probabilities
        
        Returns:
            prediction: Predicted attack type (string)
            confidence: Confidence score (float)
            probabilities: (optional) All class probabilities (dict)
        """
        # Handle 1D input (single time step)
        if sequence.ndim == 1:
            sequence = np.tile(sequence, (self.sequence_length, 1))
        
        # Validate shape
        if sequence.shape[0] != self.sequence_length:
            raise ValueError(f"Expected sequence length {self.sequence_length}, got {sequence.shape[0]}")
        if sequence.shape[1] != len(self.feature_cols):
            raise ValueError(f"Expected {len(self.feature_cols)} features, got {sequence.shape[1]}")
        
        # Preprocess
        sequence_normalized = self.scaler.transform(sequence)
        sequence_tensor = torch.FloatTensor(sequence_normalized).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(sequence_tensor)
            probabilities_tensor = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(logits, dim=1).item()
            confidence = probabilities_tensor[0, predicted_class].item()
        
        # Decode prediction
        attack_type = self.label_encoder.inverse_transform([predicted_class])[0]
        
        if return_probabilities:
            probs_dict = {
                self.label_encoder.classes_[i]: probabilities_tensor[0, i].item()
                for i in range(len(self.label_encoder.classes_))
            }
            return attack_type, confidence, probs_dict
        
        return attack_type, confidence
    
    def predict_batch(self, sequences):
        """
        Predict attack types for multiple sequences
        
        Args:
            sequences: numpy array of shape (batch_size, sequence_length, num_features)
        
        Returns:
            predictions: List of predicted attack types
            confidences: List of confidence scores
        """
        # Validate shape
        if sequences.ndim != 3:
            raise ValueError(f"Expected 3D array (batch, seq_len, features), got {sequences.ndim}D")
        
        # Preprocess
        batch_size, seq_len, num_features = sequences.shape
        sequences_reshaped = sequences.reshape(-1, num_features)
        sequences_normalized = self.scaler.transform(sequences_reshaped)
        sequences_normalized = sequences_normalized.reshape(batch_size, seq_len, num_features)
        sequences_tensor = torch.FloatTensor(sequences_normalized).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(sequences_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_classes = torch.argmax(logits, dim=1).cpu().numpy()
            confidences = probabilities.gather(1, torch.argmax(logits, dim=1).unsqueeze(1)).squeeze().cpu().numpy()
        
        # Decode predictions
        predictions = self.label_encoder.inverse_transform(predicted_classes)
        
        return predictions.tolist(), confidences.tolist()
    
    def get_feature_names(self):
        """Return list of required feature names"""
        return self.feature_cols
    
    def get_attack_classes(self):
        """Return list of possible attack classes"""
        return self.label_encoder.classes_.tolist()


def main():
    """Demo usage of the detector"""
    print("=" * 80)
    print("LEO SATELLITE ATTACK DETECTOR - INFERENCE DEMO")
    print("=" * 80)
    
    # Initialize detector
    detector = SatelliteAttackDetector(
        model_path='/home/ubuntu/leo_attack_detection/best_efficient_model.pt',
        data_path='/home/ubuntu/leo_attack_detection/processed_data.pkl'
    )
    
    # Load test data for demo
    print("\n[Demo] Loading test data...")
    with open('/home/ubuntu/leo_attack_detection/processed_data.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    
    sequences = data_dict['sequences']
    labels = data_dict['labels']
    label_encoder = data_dict['label_encoder']
    
    # Test on a few samples
    print("\n[Demo] Testing on sample sequences...")
    print("=" * 80)
    
    num_samples = 10
    indices = np.random.choice(len(sequences), num_samples, replace=False)
    
    correct = 0
    for i, idx in enumerate(indices):
        sequence = sequences[idx]
        true_label = label_encoder.inverse_transform([labels[idx]])[0]
        
        # Predict
        pred_label, confidence, probs = detector.predict(sequence, return_probabilities=True)
        
        # Display result
        is_correct = pred_label == true_label
        correct += int(is_correct)
        
        print(f"\nSample {i+1}:")
        print(f"  True Label:      {true_label}")
        print(f"  Predicted:       {pred_label} ({'✓' if is_correct else '✗'})")
        print(f"  Confidence:      {confidence:.2%}")
        print(f"  All Probabilities:")
        for attack_type, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            print(f"    {attack_type:12s}: {prob:.2%}")
    
    print("\n" + "=" * 80)
    print(f"Demo Accuracy: {correct}/{num_samples} ({correct/num_samples*100:.1f}%)")
    print("=" * 80)
    
    # Batch prediction demo
    print("\n[Demo] Testing batch prediction...")
    batch_sequences = sequences[indices]
    batch_predictions, batch_confidences = detector.predict_batch(batch_sequences)
    
    print(f"✓ Processed {len(batch_predictions)} sequences in batch")
    print(f"  Predictions: {batch_predictions}")
    print(f"  Confidences: {[f'{c:.2%}' for c in batch_confidences]}")
    
    print("\n" + "=" * 80)
    print("INFERENCE DEMO COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
