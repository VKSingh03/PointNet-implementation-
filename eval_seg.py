import numpy as np
import argparse

import torch
from models import seg_model
from data_loader import get_data_loader
from utils import create_dir, viz_seg, rotate_pc, write_experiment_results
import tqdm
import random


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='model_epoch_250')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/seg/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/seg/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output/segmentation')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    # additions for automating the code. 
    parser.add_argument('--main_dir', type=str, default='./data/')
    parser.add_argument('--task', type=str, default="cls", help='The task: cls or seg')
    parser.add_argument('--batch_size', type=int, default=8, help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=4, help='The number of threads to use for the DataLoader.')
    parser.add_argument('--rotate', type=int, default=0)
    parser.add_argument('--x', type=int, default=0)
    parser.add_argument('--y', type=int, default=0)
    parser.add_argument('--z', type=int, default=0)

    return parser

# Function to calculate batch accuracy
def calculate_batch_accuracy(pred_labels, true_labels):
    return pred_labels.eq(true_labels.data).cpu().sum().item() / true_labels.view([-1, 1]).size()[0]

# Function to visualize segmentation results
def visualize_segmentation_results(data, labels, path, sample_num, is_ground_truth, device, num_points):
    verts = data.detach().cpu()
    seg_labels = labels.detach().cpu().data

    viz_seg(verts=verts, labels=seg_labels, path=path.format(sample_num), device=device, points=num_points, viz="seg")

def evaluate_and_visualize(args, model, test_dataloader, ind):
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
        batch_labels = batch_labels[:, ind].to(args.device).to(torch.long)

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
    write_experiment_results('segmentation_experiment_results.txt', args.exp_name, args.num_points,
                             [args.x, args.y, args.z], test_accuracy)

    num_examples = 5
    # Visualize Incorrect Predictions
    visualize_results(args, test_data, test_label, pred_label,  num_examples, incorrect=True)
    # Visualize Correct Predictions
    visualize_results(args, test_data, test_label, pred_label,  num_examples, incorrect=False)


def visualize_results(args, test_data, test_label, pred_label,  num_examples, incorrect=True):
    percentage = 30 if incorrect else 80
    points_to_match = int(0.01 * percentage * pred_label.size(1))

    prediction_indices = torch.nonzero((pred_label.cpu() != test_label.cpu()).sum(dim=1) > points_to_match).squeeze() \
        if incorrect else torch.nonzero((pred_label.cpu() == test_label.cpu()).sum(dim=1) >= points_to_match).squeeze()

    prediction_indices = prediction_indices.tolist()
    if len(prediction_indices) > num_examples:
        prediction_indices = random.sample(prediction_indices, num_examples)

    sample_num = 1
    for idx in prediction_indices:
        verts = test_data[idx].detach().cpu()
        gt_seg = test_label[idx].detach().cpu().data
        pred_seg = pred_label[idx].detach().cpu().data
        result_type = "incorrect" if incorrect else "correct"

        visualize_segmentation_results(verts, gt_seg, path=f"{args.output_dir}/{args.exp_name}/{result_type}/gt_sample_{{}}.gif", 
                                       sample_num=sample_num, is_ground_truth=True, device=args.device, num_points=args.num_points)

        visualize_segmentation_results(verts, pred_seg, path=f"{args.output_dir}/{args.exp_name}/{result_type}/pred_sample_{{}}.gif", 
                                       sample_num=sample_num, is_ground_truth=False, device=args.device, num_points=args.num_points)

        sample_num += 1

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Segmentation Task  ------
    create_dir(args.output_dir+'/'+args.exp_name+'/correct')
    create_dir(args.output_dir+'/'+args.exp_name+'/incorrect')
    model = seg_model().to(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/seg/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))

    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)

    # ------ TO DO: Make Prediction ------
    # Load test data for evaluation
    test_dataloader = get_data_loader(args=args,train=False)
    evaluate_and_visualize(args, model, test_dataloader, ind)
