"""
Command line interface (CLI) execution definitions for initiating a model training. Allows me to execute
model training with variable parameters remotely on my school's on-prem GPU server by using an SSH connection.
"""
import argparse

from definitions import QUESTION_LIST
from src.architectures.random_forest.main import random_forest_experiment
from src.architectures.encoder_decoder_attentional_gru.main import (
    encoder_decoder_attentional_gru_experiment,
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "architecture",
        default="gru",
        choices=["gru", "lstm", "random_forest"],
        type=str,
        help="Model to train.",
    )
    parser.add_argument(
        "name",
        default=f"FULLSET",
        type=str,
        help="Experiment name used for saving results.",
    )
    parser.add_argument(
        "difficulty",
        default="easy",
        type=str,
        choices=["easy", "medium", "hard", "all"],
        help="Difficulty of training dataset to use.",
    )
    parser.add_argument(
        "-ne",
        "--number_epochs",
        default=150,
        type=int,
        help="Number of training epochs to perform.",
    )
    parser.add_argument(
        "-eh",
        "--encoder_hidden_size",
        default=2048,
        choices=[64, 128, 256, 512, 1024, 2048],
        type=int,
        help="Encoder hidden size selection.",
    )
    parser.add_argument(
        "-tss",
        "--train_set_size",
        default=None,
        type=int,
        help="Number of questions to train with. If none defaults to whole set.",
    )
    parser.add_argument(
        "-dh",
        "--decoder_hidden_size",
        default=2048,
        choices=[64, 128, 256, 512, 1024, 2048],
        type=int,
        help="Decoder hidden size selection.",
    )
    parser.add_argument(
        "-qt",
        "--question_type",
        default=QUESTION_LIST[0],
        choices=QUESTION_LIST,
        type=str,
        help="Question type to train algorithm on.",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        default=512,
        choices=[16, 32, 64, 128, 256, 512, 1024],
        type=int,
        help="Batch size for training.",
    )
    parser.add_argument(
        "-id",
        default="1",
        choices=["0", "1", "2", "3"],
        type=str,
        help="Cuda visible device id for use.",
    )
    parser.add_argument(
        "-co",
        "--character_offset",
        action="store_true",
        help=(
            """Flag for whether to perform top 4 character offsetting for neural networks during prediction testing. 
        See publication for technical definition."""
        ),
    )
    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="Save model pickle after training.",
    )
    parser.add_argument(
        "-a",
        "--analyzer",
        default="char",
        choices=["char", "word"],
        type=str,
        help="Vectorization scheme to use (only applicable to random forest option).",
    )
    parser.add_argument(
        "-md",
        "--max_depth",
        default=6,
        type=int,
        help="Max branch depth (only applicable to random forest option).",
    )
    parser.add_argument(
        "-nes",
        "--number_estimators",
        default=10,
        type=int,
        help="Number of trees in the random forest (only applicable to random forest option).",
    )

    args = parser.parse_args()

    if args.architecture == "random_forest":
        random_forest_experiment(
            experiment_name=args.name,
            train_set_size=args.train_set_size,
            difficulty=args.difficulty,
            question_type=args.question_type,
            analyzer=args.analyzer,
            max_depth=args.max_depth,
            number_estimators=args.number_estimators,
            save=args.save
        )

    elif args.architecture == "gru":
        encoder_decoder_attentional_gru_experiment(
            experiment_name=args.name,
            train_set_size=args.train_set_size,
            train_epochs=args.number_epochs,
            train_batch_size=args.batch_size,
            difficulty=args.difficulty,
            question_type=args.question_type,
            encoder_hidden_size=args.encoder_hidden_size,
            decoder_hiden_size=args.decoder_hidden_size,
            cuda_id=args.id,
            character_offset=args.character_offset,
            save_model=args.save,
        )

    else:
        raise NotImplementedError
