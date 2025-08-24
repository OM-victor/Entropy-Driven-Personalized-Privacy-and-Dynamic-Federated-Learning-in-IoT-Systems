import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Federated Learning with Personalized Differential Privacy and Dynamic Compression")

    parser.add_argument('--num_clients', type=int, default=11, help="Number of clients")
    parser.add_argument('--local_epoch', type=int, default=20, help="Number of local epochs")
    parser.add_argument('--global_epoch', type=int, default=100, help="Number of global epochs")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")

    parser.add_argument('--user_sample_rate', type=float, default=1, help="Sample rate for user sampling")

    parser.add_argument('--target_epsilon', type=float, default=2.5, help="Total privacy budget epsilon")
    parser.add_argument('--target_delta', type=float, default=1e-5, help="Privacy budget delta")
    parser.add_argument('--clipping_bound', type=float, default=1.0, help="Gradient clipping bound")

    parser.add_argument('--fisher_threshold', type=float, default=0.4, help="Fisher information threshold for parameter selection")
    parser.add_argument('--lambda_1', type=float, default=0.1, help="Lambda value for EWC regularization term")
    parser.add_argument('--lambda_2', type=float, default=0.05, help="Lambda value for regularization term to control the update magnitude")

    parser.add_argument('--device', type=int, default=0, help='Set the visible CUDA device for calculations')

    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")

    parser.add_argument('--no_clip', action='store_true', help="Disable gradient clipping")
    parser.add_argument('--no_noise', action='store_true', help="Disable noise addition")

    parser.add_argument('--dataset', type=str, default='CIFAR10', help="Dataset name (MNIST/FashionMNIST/CIFAR10)")

    parser.add_argument('--dir_alpha', type=float, default=100, help="Dirichlet distribution alpha for data heterogeneity")

    parser.add_argument('--dirStr', type=str, default='', help="Directory string for logging")

    parser.add_argument('--store', action='store_true', help="Store results to file")

    parser.add_argument('--appendix', type=str, default='', help="Appendix string")

    parser.add_argument('--adaptive_dp', action='store_true', help="Use adaptive personalized differential privacy")
    parser.add_argument('--adaptation_rate', type=float, default=0.1, help="Adaptation rate for adaptive DP")

    parser.add_argument('--compression', action='store_true', help="Enable model update compression")
    parser.add_argument('--residual', action='store_true', help="Enable residual mechanism")

    parser.add_argument('--dynamic_compression', action='store_true', help="Enable dynamic weight compression")
    parser.add_argument('--c_max', type=float, default=0.9999, help="Maximum compression ratio (99.99%)")
    parser.add_argument('--c_min', type=float, default=0.99, help="Minimum compression ratio (99.00%)")
    parser.add_argument('--compression_sigma', type=float, default=10.0, help="Compression sensitivity parameter σ")
    parser.add_argument('--w_a', type=float, default=0.5, help="Weight for accuracy in performance change calculation")
    parser.add_argument('--w_b', type=float, default=0.5, help="Weight for loss in performance change calculation")
    parser.add_argument('--residual_threshold', type=float, default=0.1, help="Threshold for residual accumulation")
    parser.add_argument('--sensitivity_analysis', action='store_true', help="Enable sensitivity analysis for compression parameters")

    args = parser.parse_args()
    return args
