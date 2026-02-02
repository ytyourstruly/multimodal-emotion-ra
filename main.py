"""
Main training script with integrated CREMA-D support
Supports both RAVDESS and CREMA-D datasets with k-fold cross-validation
"""
import os
import json
import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
import pickle

from opts import parse_opts
from model import generate_model
import transforms 
from dataset import get_training_set, get_validation_set, get_test_set
from utils import Logger, adjust_learning_rate, save_checkpoint, plot_confusion_matrix, report, save_csv_report, save_confusion_matrix, get_emotion_labels
from train import train_epoch
from validation import val_epoch
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
# from inference import TrainingAlignedInference




if __name__ == '__main__':
    opt = parse_opts()
    n_folds = 5

    # Setup device
    if opt.device != 'cpu':
        opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if not opt.only_inference:
        pretrained = opt.pretrain_path != 'None'    
        
        if not os.path.exists(opt.result_path):
            os.makedirs(opt.result_path)
        
        # ========== CREMA-D SETUP ========== transferred to their own file handling
        
        opt.arch = '{}'.format(opt.model)  
        opt.store_name = '_'.join([opt.dataset, opt.model, str(opt.sample_duration)])
        
        # Get emotion labels for this dataset
        emotion_labels = get_emotion_labels(opt.dataset)
        
        # ========== K-FOLD TRAINING ==========
        for fold in range(n_folds):
            print(f"\n{'='*70}")
            print(f"FOLD {fold+1}/{n_folds}")
            print(f"{'='*70}\n")
            
            train_loss_history = {}
            train_acc_history = {}
            val_loss_history = {}
            val_acc_history = {}
            
            # Set annotation path based on dataset
            if opt.dataset == 'RAVDESS':
                opt.annotation_path += '/annotations_croppad_fold' + str(fold+1) + '.txt'
            elif opt.dataset == 'CREMAD':
                opt.annotation_path += f'/annotations_croppad_fold{fold+1}.txt'
            
            print(f"Using annotation file: {opt.annotation_path}")
            print(opt)
            

            
            torch.manual_seed(opt.manual_seed)
            model, parameters = generate_model(opt)

            criterion = nn.CrossEntropyLoss().to(opt.device)
            
            # -------- TRAINING SETUP --------
            if not opt.no_train:
                video_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotate(),
                    transforms.ToTensor(opt.video_norm_value)])

                training_data = get_training_set(opt, spatial_transform=video_transform) 
            
                train_loader = torch.utils.data.DataLoader(
                    training_data,
                    batch_size=opt.batch_size,
                    shuffle=True,
                    num_workers=opt.n_threads,
                    pin_memory=True)

                optimizer = optim.SGD(
                    parameters,
                    lr=opt.learning_rate,
                    momentum=opt.momentum,
                    dampening=opt.dampening,
                    weight_decay=opt.weight_decay,
                    nesterov=False)

                scheduler = lr_scheduler.ReduceLROnPlateau(
                    optimizer, 'min', patience=opt.lr_patience)
            
            # -------- VALIDATION SETUP --------
            if not opt.no_val:
                video_transform = transforms.Compose([
                    transforms.ToTensor(opt.video_norm_value)])     

                validation_data = get_validation_set(opt, spatial_transform=video_transform)
                
                val_loader = torch.utils.data.DataLoader(
                    validation_data,
                    batch_size=opt.batch_size,
                    shuffle=False,
                    num_workers=opt.n_threads,
                    pin_memory=True)

            best_prec1 = 0
            best_loss = 1e10

            # Resume from checkpoint if specified
            if opt.resume_path:
                print('loading checkpoint {}'.format(opt.resume_path))
                checkpoint = torch.load(opt.resume_path, weights_only=False)
                assert opt.arch == checkpoint['arch']
                best_prec1 = checkpoint['best_prec1']
                opt.begin_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])

            # -------- TRAINING LOOP --------
            for i in range(opt.begin_epoch, opt.n_epochs + 1):
                # -------- TRAIN --------
                if not opt.no_train:
                    adjust_learning_rate(optimizer, i, opt)
                    train_loss, train_acc = train_epoch(i, train_loader, model, criterion, optimizer, opt)

                    train_loss_history[i] = float(train_loss)
                    train_acc_history[i] = float(train_acc)

                    scheduler.step(train_loss)

                    state = {
                        'epoch': i,
                        'arch': opt.arch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_prec1': best_prec1
                    }
                    save_checkpoint(state, False, opt, fold)
                
                # -------- VALIDATION --------
                if not opt.no_val:
                    val_loss, prec1, y_true, y_pred = val_epoch(i, val_loader, model, criterion, opt)

                    val_loss_history[i] = float(val_loss)
                    val_acc_history[i] = float(prec1)

                    is_best = prec1 > best_prec1
                    best_prec1 = max(prec1, best_prec1)

                    state = {
                        'epoch': i,
                        'arch': opt.arch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer,
                        'best_prec1': best_prec1
                    }
                
                    save_checkpoint(state, is_best, opt, fold)
                    train_loss_path = "train_loss_" + str(fold+1) + ".pkl"
                    with open(os.path.join(opt.result_path, train_loss_path), "wb") as f:
                        pickle.dump(train_loss_history, f)
                    train_acc_path = "train_acc_" + str(fold+1) + ".pkl"
                    with open(os.path.join(opt.result_path, train_acc_path), "wb") as f:
                        pickle.dump(train_acc_history, f)
                    val_loss_path = "val_loss_" + str(fold+1) + ".pkl"
                    with open(os.path.join(opt.result_path, val_loss_path), "wb") as f:
                        pickle.dump(val_loss_history, f)
                    val_acc_path = "val_acc_" + str(fold+1) + ".pkl"
                    with open(os.path.join(opt.result_path, val_acc_path), "wb") as f:
                        pickle.dump(val_acc_history, f)
                    # Save options
                    with open(os.path.join(opt.result_path, 'opts'+str(time.time())+str(fold)+'.json'), 'w') as opt_file:
                        json.dump(vars(opt), opt_file)
                    # Save training history
            # -------- TESTING --------
            if opt.test:
                print(f"\n{'='*70}")
                print(f"TESTING FOLD {fold+1}")
                print(f"{'='*70}\n")
                
                base_dir = opt.result_path
                test_accuracies = []

                video_transform = transforms.Compose([
                    transforms.ToTensor(opt.video_norm_value)
                ])

                test_data = get_test_set(opt, spatial_transform=video_transform)

                # Create folder for this fold
                fold_dir = os.path.join(base_dir, f"fold_{fold+1}")
                os.makedirs(fold_dir, exist_ok=True)

                # Load best model for this fold
                best_model_path = os.path.join(
                    base_dir,
                    f"{opt.store_name}_best{fold}.pth"
                )
                
                if not os.path.exists(best_model_path):
                    print(f"Warning: Best model not found at {best_model_path}")
                    print("Skipping testing for this fold")
                    continue
                
                best_state = torch.load(best_model_path, weights_only=False)
                model.load_state_dict(best_state["state_dict"])

                test_loader = torch.utils.data.DataLoader(
                    test_data,
                    batch_size=opt.batch_size,
                    shuffle=False,
                    num_workers=opt.n_threads,
                    pin_memory=True
                )

                test_loss, test_prec1, y_true, y_pred = val_epoch(
                    10000, test_loader, model, criterion, opt
                )

                # Save results
                save_csv_report(y_true, y_pred, fold_dir, fold, emotion_labels)
                save_confusion_matrix(y_true, y_pred, fold_dir, fold, emotion_labels)
                
                with open(os.path.join(fold_dir, f"test_summary_fold_{fold+1}.txt"), "w") as f:
                    f.write(f"Dataset: {opt.dataset}\n")
                    f.write(f"Fold: {fold+1}\n")
                    f.write(f"Precision@1: {test_prec1:.4f}\n")
                    f.write(f"Loss: {test_loss:.4f}\n")
                
                test_accuracies.append(test_prec1)
                print(f"Fold {fold+1} Test Accuracy: {test_prec1:.4f}")
        
        # -------- SUMMARY ACROSS ALL FOLDS --------
        if opt.test and len(test_accuracies) > 0:
            print(f"\n{'='*70}")
            print(f"FINAL RESULTS - {opt.dataset}")
            print(f"{'='*70}")
            print(f"Mean Accuracy: {np.mean(test_accuracies):.4f} ± {np.std(test_accuracies):.4f}")
            print(f"All fold accuracies: {test_accuracies}")
            
            summary_file = os.path.join(opt.result_path, 'test_summary.txt')
            with open(summary_file, 'w') as f:
                f.write(f"Dataset: {opt.dataset}\n")
                f.write(f"Model: {opt.model}\n")
                f.write(f"Number of folds: {n_folds}\n")
                f.write(f"Mean Accuracy: {np.mean(test_accuracies):.4f} ± {np.std(test_accuracies):.4f}\n")
                f.write(f"Individual fold accuracies:\n")
                for i, acc in enumerate(test_accuracies):
                    f.write(f"  Fold {i+1}: {acc:.4f}\n")
            
            print(f"\nSummary saved to: {summary_file}")
    
    # -------- INFERENCE MODE --------
    if opt.media_path != "":
        print(f"\n{'='*70}")
        print(f"INFERENCE MODE")
        print(f"{'='*70}\n")
        
        model_path = opt.result_path
        model_paths = [
            os.path.join(model_path, f"fold_{i+1}", f"{opt.store_name}_best{i}.pth")
            for i in range(n_folds)
        ]
        results = []

        for i, mp in enumerate(model_paths):
            if not os.path.exists(mp):
                print(f"Warning: Model {i+1} not found at {mp}")
                continue
                
            print(f"\n=== Running Model {i+1}: {mp}")
            
            infer = TrainingAlignedInference()
            infer.device = torch.device(opt.device)
            infer.model.to(infer.device)

            out = infer.inference_sliding_window(
                opt.media_path,
                step_sec=1.0,
                plot_results=False,
                model_name=f"Model {i+1}"
            )

            results.append(out)

        # Plot comparison of all models
        if len(results) > 0:
            TrainingAlignedInference.plot_unified_models_grid(results)
            print(f"\nInference complete for {len(results)} models")
        else:
            print("No models found for inference")