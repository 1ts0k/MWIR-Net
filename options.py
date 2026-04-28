import argparse

parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--cuda', type=int, default=0)

parser.add_argument('--epochs', type=int, default=120, help='maximum number of epochs to train the total model.')
parser.add_argument('--batch_size', type=int,default=8,help="Batch size to use per GPU")
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate of encoder.')
parser.add_argument('--warmup_epochs', type=int, default=2,
                    help='number of warmup epochs for the learning rate scheduler.')
parser.add_argument('--max_steps', type=int, default=-1,
                    help='optional trainer max_steps for quick smoke tests; -1 means full epoch training.')

parser.add_argument('--de_type', nargs='+', default=['derain', 'dehaze'],
                    help='which type of degradations is training and testing for.')

parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')
parser.add_argument('--num_workers', type=int, default=8, help='number of workers.')

# path
parser.add_argument('--data_file_dir', type=str, default='data_dir/',  help='where clean images of denoising saves.')
parser.add_argument('--denoise_dir', type=str, default='data/Train/Denoise/',
                    help='where clean images of denoising saves.')
parser.add_argument('--derain_dir', type=str, default='data/Train/Derain/',
                    help='where training images of deraining saves.')
parser.add_argument('--dehaze_dir', type=str, default='data/Train/Dehaze/',
                    help='where training images of dehazing saves.')
parser.add_argument('--derain_repeat', type=int, default=1,
                    help='repeat factor for deraining ids. Official Rain100L used a large repeat; RAIN13K should usually stay at 1.')
parser.add_argument('--max_derain', type=int, default=0,
                    help='maximum number of deraining samples to use after repeat; 0 means use all.')
parser.add_argument('--max_dehaze', type=int, default=0,
                    help='maximum number of dehazing samples to use; 0 means use all.')
parser.add_argument('--subset_seed', type=int, default=0,
                    help='seed for reproducible random subsampling when max_derain/max_dehaze is set.')
parser.add_argument('--edge_loss_weight', type=float, default=0.05,
                    help='weight for Sobel edge consistency loss; set 0 to disable.')
parser.add_argument('--pixel_loss_type', type=str, default='l1', choices=['l1', 'charbonnier'],
                    help='pixel reconstruction loss type.')
parser.add_argument('--charbonnier_eps', type=float, default=1e-3,
                    help='epsilon used by Charbonnier loss.')
parser.add_argument('--init_ckpt', type=str, default=None,
                    help='optional checkpoint for compatible weight initialization before training.')
parser.add_argument('--ablation_mode', type=str, default='full',
                    choices=['full', 'zero_prompt', 'no_channel_attention'],
                    help='MWIR-Net module ablation mode.')
parser.add_argument('--output_path', type=str, default="output/", help='output save path')
parser.add_argument('--ckpt_path', type=str, default="ckpt/Denoise/", help='checkpoint save path')
parser.add_argument("--wblogger",type=str,default=None,help = "W&B project name. Use none/false/off or omit to log locally.")
parser.add_argument("--ckpt_dir",type=str,default="train_ckpt",help = "Name of the Directory where the checkpoint is to be saved")
parser.add_argument("--num_gpus",type=int,default=1,help = "Number of GPUs to use for training")
parser.add_argument("--precision", type=str, default="32-true", help="Lightning precision setting, e.g. 32-true or 16-mixed")

options = parser.parse_args()
