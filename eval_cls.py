# import imp
import numpy as np
import argparse

import torch
from models import cls_model
from utils import create_dir
import tqdm
from utils import viz_seg, rotate_pc, write_experiment_results
from data_loader import get_data_loader
import random

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='model_epoch_250')
    parser.add_argument('--i', type=int, default=1, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output/classification')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')
    
    # Parameters useful for automating the experiments. 
    parser.add_argument('--rotate', type=int, default=0)
    parser.add_argument('--x', type=int, default=0)
    parser.add_argument('--y', type=int, default=0)
    parser.add_argument('--z', type=int, default=0)
    parser.add_argument('--class_num', type=int, default=0)
    parser.add_argument('--main_dir', type=str, default='./data/')
    parser.add_argument('--task', type=str, default="cls", help='The task: cls or seg')
    parser.add_argument('--batch_size', type=int, default=8, help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')

    return parser

def calculate_batch_accuracy(pred_labels, true_labels):
    return pred_labels.eq(true_labels.data).cpu().sum().item() / true_labels.size()[0]

def visualize_classification_results(data, label, pred_label, class_name, path, ind, device):
    verts = data.detach().cpu()
    gt_cls = label.detach().cpu().data
    pred_cls = pred_label.detach().cpu().data

    viz_seg(verts=verts, path=path.format(ind), class_name=class_name[pred_cls], device=device, viz="cls")

def evaluate_and_visualize_classification(args, model, test_dataloader, ind):
    # Evaluation
    pred_label = []
    test_label = []
    test_data = []

    batch_num = 1
    for batch in test_dataloader:
        batch_testdata, batch_labels = batch
        if args.rotate:
            batch_testdata = rotate_pc(batch_testdata, [args.x, args.y, args.z])
        batch_testdata = batch_testdata[:, ind].to(args.device)
        batch_labels = batch_labels.to(args.device).to(torch.long)

        with torch.no_grad():
            batch_pred_labels = torch.argmax(model(batch_testdata), dim=-1, keepdim=False)

        batch_test_accuracy = calculate_batch_accuracy(batch_pred_labels, batch_labels)
        print(f"Batch {batch_num}: Test Accuracy {batch_test_accuracy}")
        batch_num += 1

        pred_label.append(batch_pred_labels)
        test_label.append(batch_labels)
        test_data.append(batch_testdata)

    pred_label = torch.cat(pred_label, dim=0)
    test_label = torch.cat(test_label, dim=0)
    test_data = torch.cat(test_data, dim=0)

    test_accuracy = calculate_batch_accuracy(pred_label, test_label)
    print("Test accuracy: {}".format(test_accuracy))
    # Write experiment results to file
    class_name = ["chair", "vase", "lamp"]
    write_experiment_results('classification_experiment_results.txt', args.exp_name, 
                             args.num_points, [args.x, args.y, args.z], test_accuracy, class_name=class_name[args.class_num])

    num_examples = 1
    # Visualize Incorrect Predictions
    visualize_results_classification(args, test_data, test_label, pred_label, class_name, num_examples, incorrect=True)
    # Visualize Correct Predictions
    visualize_results_classification(args, test_data, test_label, pred_label, class_name, num_examples, incorrect=False)


def visualize_results_classification(args, test_data, test_label, pred_label, class_name, num_examples, incorrect=True):
    desired_class = args.class_num

    indices = torch.nonzero((pred_label.cpu() == desired_class) & (pred_label.cpu() != test_dataloader.dataset.label)).squeeze() \
        if incorrect else torch.nonzero((pred_label.cpu() == desired_class) & (pred_label.cpu() == test_dataloader.dataset.label)).squeeze()

    indices = indices.tolist()
    indices = random.sample(indices, num_examples)

    result_type = "incorrect" if incorrect else "correct"
    for ind in indices:
        visualize_classification_results(test_data[ind], test_label[ind], pred_label[ind], class_name,
                                         f"output/classification/{args.exp_name}/{result_type}/classification_{{}}_gt_{class_name[test_label[ind]]}_pred_{class_name[pred_label[ind]]}.gif", ind, device="cuda")

if __name__ == '__main__':
    
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Classification Task ------
    model = cls_model().to(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/cls/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))

    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)

    # ------ TO DO: Make Prediction ------
    # Creating folders for storing results 
    create_dir(args.output_dir+'/'+args.exp_name+'/correct')
    create_dir(args.output_dir+'/'+args.exp_name+'/incorrect')

    # Loading test data for evaluation.
    test_dataloader = get_data_loader(args=args,train=False)

    # Calling evaluate and visualize function
    evaluate_and_visualize_classification(args, model, test_dataloader, ind)