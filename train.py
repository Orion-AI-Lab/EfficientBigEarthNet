import argparse
import json
import os
import random as rn
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

import models

# from gradcam import GradCAM
from inputs import create_batched_dataset
from metrics import CustomMetrics
from models import MODELS_CLASS

SEED = 42


def evaluate_model(model, batched_dataset, nb_class, args, gradcam=False):
    """Evaluate model using the eval dataset"""
    print("Running model on validation dataset")
    custom_metrics = CustomMetrics(nb_class=nb_class)

    processed_imgs = 0
    start = time.time()
    for _, single_batch in enumerate(iter(batched_dataset)):
        if args["modality"] == "S1":
            x_all = [
                single_batch["VV"],
                single_batch["VH"],
            ]
        elif args["modality"] == "S2":
            x_all = [
                single_batch["B02"],
                single_batch["B03"],
                single_batch["B04"],
                single_batch["B05"],
                single_batch["B06"],
                single_batch["B07"],
                single_batch["B08"],
                single_batch["B8A"],
                single_batch["B11"],
                single_batch["B12"],
            ]
        elif args["modality"] == "MM":
            x_all = [
                single_batch["B02"],
                single_batch["B03"],
                single_batch["B04"],
                single_batch["B05"],
                single_batch["B06"],
                single_batch["B07"],
                single_batch["B08"],
                single_batch["B8A"],
                single_batch["B11"],
                single_batch["B12"],
                single_batch["VV"],
                single_batch["VH"],
            ]
        y = single_batch[args["label_type"] + "_labels_multi_hot"]
        # Compare predicted label to actual label
        y_ = model(x_all, training=False)

        # Update all custom metrics
        custom_metrics.update_state(y, y_)

        processed_imgs += x_all[0].shape[0]

        if gradcam:
            gradcam_x_all = []
            for j in range(len(single_batch)):
                if args["modality"] == "S1":
                    gradcam_x_all.append(
                        [
                            single_batch["VV"][j : j + 1][:][:],
                            single_batch["VH"][j : j + 1][:][:],
                        ]
                    )
                elif args["modality"] == "S2":
                    gradcam_x_all.append(
                        [
                            single_batch["B02"][j : j + 1][:][:],
                            single_batch["B03"][j : j + 1][:][:],
                            single_batch["B04"][j : j + 1][:][:],
                            single_batch["B05"][j : j + 1][:][:],
                            single_batch["B06"][j : j + 1][:][:],
                            single_batch["B07"][j : j + 1][:][:],
                            single_batch["B08"][j : j + 1][:][:],
                            single_batch["B8A"][j : j + 1][:][:],
                            single_batch["B11"][j : j + 1][:][:],
                            single_batch["B12"][j : j + 1][:][:],
                        ]
                    )
                elif args["modality"] == "MM":
                    gradcam_x_all.append(
                        [
                            single_batch["B02"][j : j + 1][:][:],
                            single_batch["B03"][j : j + 1][:][:],
                            single_batch["B04"][j : j + 1][:][:],
                            single_batch["B05"][j : j + 1][:][:],
                            single_batch["B06"][j : j + 1][:][:],
                            single_batch["B07"][j : j + 1][:][:],
                            single_batch["B08"][j : j + 1][:][:],
                            single_batch["B8A"][j : j + 1][:][:],
                            single_batch["B11"][j : j + 1][:][:],
                            single_batch["B12"][j : j + 1][:][:],
                            single_batch["VV"][j : j + 1][:][:],
                            single_batch["VH"][j : j + 1][:][:],
                        ]
                    )

            if args["modality"] == "S1":
                patch_names = single_batch["patch_name_s1"].values.numpy().tolist()
            elif args["modality"] == "S2":
                patch_names = single_batch["patch_name_s2"].values.numpy().tolist()
            elif args["modality"] == "MM":
                s1 = single_batch["patch_name_s1"].values.numpy().tolist()
                s2 = single_batch["patch_name_s2"].values.numpy().tolist()
                patch_names = list(map("-".join, zip(s1, s2)))
            else:
                raise ValueError("Incompatible modality.")

            for index, x in enumerate(gradcam_x_all):
                GradCAM(model, x, y[index], patch_names[index])

    if args["worker_index"] == 0:
        print(f"Inference rate: {processed_imgs // (time.time() - start)} images/sec")

    (
        micro_precision,
        macro_precision,
        micro_recall,
        macro_recall,
        micro_accuracy,
        macro_accuracy,
        micro_f1score,
        macro_f1score,
        micro_f2score,
        macro_f2score,
    ) = custom_metrics.result()

    if args["parallel"]:
        print(
            f"Time : {str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))} Process {hvd.rank()} Reduce",
            flush=True,
        )
        hvd.join()
        micro_precision = hvd.allreduce(micro_precision)
        macro_precision = hvd.allreduce(macro_precision)
        micro_recall = hvd.allreduce(micro_recall)
        macro_recall = hvd.allreduce(macro_recall)
        micro_accuracy = hvd.allreduce(micro_accuracy)
        macro_accuracy = hvd.allreduce(macro_accuracy)
        micro_f1score = hvd.allreduce(micro_f1score)
        macro_f1score = hvd.allreduce(macro_f1score)
        micro_f2score = hvd.allreduce(micro_f2score)
        macro_f2score = hvd.allreduce(macro_f2score)

    return (
        micro_precision,
        macro_precision,
        micro_recall,
        macro_recall,
        micro_accuracy,
        macro_accuracy,
        micro_f1score,
        macro_f1score,
        micro_f2score,
        macro_f2score,
    )


def _write_summary(summary_writer, custom_metrics, epoch):
    (
        epoch_micro_precision,
        epoch_macro_precision,
        epoch_micro_recall,
        epoch_macro_recall,
        epoch_micro_accuracy,
        epoch_macro_accuracy,
        epoch_micro_f1score,
        epoch_macro_f1score,
        epoch_micro_f2score,
        epoch_macro_f2score,
    ) = custom_metrics

    with summary_writer.as_default():
        tf.summary.scalar("micro_precision", epoch_micro_precision, step=epoch)
        tf.summary.scalar("macro_precision", epoch_macro_precision, step=epoch)
        tf.summary.scalar("micro_recall", epoch_micro_recall, step=epoch)
        tf.summary.scalar("macro_recall", epoch_macro_recall, step=epoch)
        tf.summary.scalar("micro_accuracy", epoch_micro_accuracy, step=epoch)
        tf.summary.scalar("macro_accuracy", epoch_macro_accuracy, step=epoch)
        tf.summary.scalar("micro_f1score", epoch_micro_f1score, step=epoch)
        tf.summary.scalar("macro_f1score", epoch_macro_f1score, step=epoch)
        tf.summary.scalar("micro_f2score", epoch_micro_f2score, step=epoch)
        tf.summary.scalar("macro_f2score", epoch_macro_f2score, step=epoch)


def run_model(args):
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Eager execution: {tf.executing_eagerly()}")
    print(f"Running using random seed: {SEED}")
    print(f"Batch size: {args['batch_size']}")
    print(f"Epochs: {args['nb_epoch']}")

    rn.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # Create our data pipeline
    train_batched_dataset = create_batched_dataset(
        args["tr_tf_record_files"],
        args["batch_size"],
        args["shuffle_buffer_size"],
        args["label_type"],
        args["num_workers"],
        args["worker_index"],
    )

    val_batched_dataset = create_batched_dataset(
        args["val_tf_record_files"],
        args["batch_size"],
        args["shuffle_buffer_size"],
        args["label_type"],
        args["num_workers"],
        args["worker_index"],
    )

    test_batched_dataset = create_batched_dataset(
        args["test_tf_record_files"],
        args["batch_size"],
        args["shuffle_buffer_size"],
        args["label_type"],
        args["num_workers"],
        args["worker_index"],
    )

    # Create our model
    nb_class = 19 if args["label_type"] == "BigEarthNet-19" else 43

    try:
        bigearth_model_class = MODELS_CLASS[args["model_name"]]
    except:
        bigearth_model_class = MODELS_CLASS["dense"]

    print(f"Creating model: {args['model_name']}")
    if args["model_name"] in ("EfficientNet", "WideResNet"):
        bigearth_model = getattr(models, bigearth_model_class)(
            nb_class=nb_class, modality=args["modality"], coefficients=args["hparams"]
        )
    else:
        bigearth_model = getattr(models, bigearth_model_class)(
            nb_class=nb_class, modality=args["modality"]
        )
    model = bigearth_model.model
    if args["worker_index"] == 0:
        print(model.summary())

    # DEBUG (use this to understand what the iterators are returning)
    # debug = False
    # if debug:
    #     single_batch = next(iter(train_batched_dataset))
    #     x_all = [
    #         single_batch["B02"],
    #         single_batch["B03"],
    #         single_batch["B04"],
    #         single_batch["B05"],
    #         single_batch["B06"],
    #         single_batch["B07"],
    #         single_batch["B08"],
    #         single_batch["B8A"],
    #         single_batch["B11"],
    #         single_batch["B12"],
    #         single_batch["VV"],
    #         single_batch["VH"],
    #     ]
    #     y = single_batch[args["label_type"] + "_labels_multi_hot"]
    #     y_ = model(x_all, training=False)

    # Create loss
    loss = tf.keras.losses.BinaryCrossentropy(
        label_smoothing=tf.cast(args["label_smoothing"], tf.float64)
    )
    print(f"Learning Rate : {args['learning_rate']}")
    # Setup training step
    @tf.function
    def training_step(inputs, targets, first_batch):
        with tf.GradientTape() as tape:
            y_pred = model(inputs, training=True)
            loss_value = loss(y_true=targets, y_pred=y_pred)

        if args["parallel"]:
            # Horovod: add Horovod Distributed GradientTape.
            tape = hvd.DistributedGradientTape(tape)

        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if args["parallel"]:
            if first_batch:
                hvd.broadcast_variables(model.variables, root_rank=0)
                hvd.broadcast_variables(optimizer.variables(), root_rank=0)

        return loss_value, y_pred

    # Setup optimizer
    step_epochs = args["decay_step"]
    nb_iterations_per_epoch = (args["training_size"] / args["num_workers"]) / args[
        "batch_size"
    ]

    decay_step = int(step_epochs * nb_iterations_per_epoch)
    back_passes = args["backward_passes"]
    decay_rate = args["decay_rate"]
    print("decay step : ", decay_step)
    print("Back passes : ", back_passes)
    print("Decay rate : ", decay_rate)
    learning_rate = args["learning_rate"] * args["num_workers"]

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if args["num_workers"] > 2:
        optimizer = hvd.DistributedOptimizer(
            optimizer, backward_passes_per_step=back_passes
        )

    # Setup metrics logging
    logdir = f"logs/scalars/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    train_summary_writer = tf.summary.create_file_writer(os.path.join(logdir, "train"))
    train_summary_writer.set_as_default()
    test_summary_writer = tf.summary.create_file_writer(os.path.join(logdir, "test"))
    checkpoint_dir = f"./checkpoint_{args['model_name']}/checkpoints"
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    # The main loop
    batch_size = args["batch_size"]
    epoch_custom_metrics = CustomMetrics(nb_class=nb_class)
    bestfscore = 0

    if args["mode"] == "eval":
        check_dir = args["eval_checkpoint"]
        checkpoint_test = tf.train.Checkpoint(model=model, optimizer=optimizer)
        checkpoint_test.restore(tf.train.latest_checkpoint(check_dir))
        test_eval = evaluate_model(model, test_batched_dataset, nb_class, args, True)
        (
            epoch_micro_precision,
            epoch_macro_precision,
            epoch_micro_recall,
            epoch_macro_recall,
            epoch_micro_accuracy,
            epoch_macro_accuracy,
            epoch_micro_f1score,
            epoch_macro_f1score,
            epoch_micro_f2score,
            epoch_macro_f2score,
        ) = test_eval
        if args["worker_index"] == 0:
            print("\n\n\n\n Test Scores \n\n\n=============")

            print(
                f"Evaluation : micro: accuracy: {epoch_micro_accuracy:.3f}, precision: {epoch_micro_precision:.3f}, recall: {epoch_micro_recall:.3f}, f1-score: {epoch_micro_f1score:.3f}, f2-score: {epoch_micro_f2score:.3f}"
            )

            print(
                f"\t macro: accuracy: {epoch_macro_accuracy:.3f}, precision: {epoch_macro_precision:.3f}, recall: {epoch_macro_recall:.3f}, f1-score: {epoch_macro_f1score:.3f}, f2-score: {epoch_macro_f2score:.3f}"
            )
        return 0

    if args["worker_index"] == 0:
        start = time.time()
    for epoch in range(args["nb_epoch"]):
        if epoch % args["decay_step"] == 0 and epoch > 0:
            optimizer.lr = optimizer.lr * args["decay_rate"]
        print(f"\nProcess {args['worker_index']} : Starting epoch {epoch}")
        print(f"Learning rate: {optimizer._decayed_lr('float32').numpy():.6f}")

        epoch_loss_avg = tf.keras.metrics.Mean(dtype="float64")
        epoch_custom_metrics.reset_states()

        nb_iterations = (args["training_size"] / args["num_workers"]) / args[
            "batch_size"
        ]
        if (args["training_size"] / args["num_workers"]) % args["batch_size"] != 0:
            nb_iterations += 1

        progress_bar = tf.keras.utils.Progbar(target=nb_iterations)

        batch_iterator = iter(train_batched_dataset)
        for i, single_batch in enumerate(batch_iterator):
            if args["modality"] == "S1":
                x_all = [
                    single_batch["VV"],
                    single_batch["VH"],
                ]
            elif args["modality"] == "S2":
                x_all = [
                    single_batch["B02"],
                    single_batch["B03"],
                    single_batch["B04"],
                    single_batch["B05"],
                    single_batch["B06"],
                    single_batch["B07"],
                    single_batch["B08"],
                    single_batch["B8A"],
                    single_batch["B11"],
                    single_batch["B12"],
                ]
            elif args["modality"] == "MM":
                x_all = [
                    single_batch["B02"],
                    single_batch["B03"],
                    single_batch["B04"],
                    single_batch["B05"],
                    single_batch["B06"],
                    single_batch["B07"],
                    single_batch["B08"],
                    single_batch["B8A"],
                    single_batch["B11"],
                    single_batch["B12"],
                    single_batch["VV"],
                    single_batch["VH"],
                ]
            y = single_batch[args["label_type"] + "_labels_multi_hot"]

            # Optimize the model
            first_batch = i == 0 and epoch == 0
            loss_value, y_ = training_step(x_all, y, first_batch)
            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss

            # Update all custom metrics
            epoch_custom_metrics.update_state(y, y_)
            if i % 20 == 0 and args["worker_index"] == 0:
                # print('Process %d Epoch %d Iteration %d'%(args['worker_index'],epoch,i),flush=True)
                print(
                    f"Process {args['worker_index']:01d}:  Epoch {epoch:03d}: Iteration {i:03d} Loss: {loss_value.numpy():.3f}"
                )
            progress_bar.update(i + 1)

        # End epoch

        if epoch % 10 == 0 or epoch == args["nb_epoch"] - 1:
            if args["worker_index"] == 0:
                eval_start = time.time()

            tf.summary.scalar("loss", epoch_loss_avg.result(), step=epoch)
            _write_summary(train_summary_writer, epoch_custom_metrics.result(), epoch)
            print(
                f"Process {args['worker_index']:01d}:  Epoch : {epoch:03d}: Loss: {epoch_loss_avg.result():.3f}"
            )
            # Evaluate model using the eval dataset
            evaluation = evaluate_model(model, val_batched_dataset, nb_class, args)

            _write_summary(test_summary_writer, evaluation, epoch)

            (
                epoch_micro_precision,
                epoch_macro_precision,
                epoch_micro_recall,
                epoch_macro_recall,
                epoch_micro_accuracy,
                epoch_macro_accuracy,
                epoch_micro_f1score,
                epoch_macro_f1score,
                epoch_micro_f2score,
                epoch_macro_f2score,
            ) = evaluation
            if args["worker_index"] == 0 and epoch_micro_f1score > bestfscore:
                bestfscore = epoch_micro_f1score.numpy()
                print(
                    f"Process {args['worker_index']:01d}: New Best F1-Score : {bestfscore:.3f}\n Epoch : {epoch:03d} Writing Checkpoint"
                )
                print(
                    f"Process {args['worker_index']:01d}:  Epoch : {epoch:03d} Writing Checkpoint"
                )
                checkpoint.save(checkpoint_dir)
                print(
                    f"Epoch {epoch:03d}: micro: accuracy: {epoch_micro_accuracy:.3f}, precision: {epoch_micro_precision:.3f}, recall: {epoch_micro_recall:.3f}, f1-score: {epoch_micro_f1score:.3f}, f2-score: {epoch_micro_f2score:.3f}"
                )
                print(
                    f"Epoch {epoch:03d}: macro: accuracy: {epoch_macro_accuracy:.3f}, precision: {epoch_macro_precision:.3f}, recall: {epoch_macro_recall:.3f}, f1-score: {epoch_macro_f1score:.3f}, f2-score: {epoch_macro_f2score:.3f}"
                )

            if args["worker_index"] == 0:
                eval_end = time.time()
                eval_time = eval_end - eval_start
                eval_hours, eval_remainder = divmod(eval_end - eval_start, 3600)
                eval_minutes, eval_seconds = divmod(eval_remainder, 60)

                start = start + eval_time
                train_end = time.time()
                train_time = train_end - start
                train_hours, train_remainder = divmod(train_time, 3600)
                train_minutes, train_seconds = divmod(train_remainder, 60)

                print(
                    f"Train took: {int(train_hours):0>2}hr. {int(train_minutes):0>2}min. {train_seconds:05.2f}sec."
                )
                print(
                    f"Validation took: {int(eval_hours):0>2}hr. {int(eval_minutes):0>2}min. {eval_seconds:05.2f}sec."
                )

            if epoch == args["nb_epoch"] - 1:
                checkpoint_test = tf.train.Checkpoint(model=model, optimizer=optimizer)
                checkpoint_test.restore(tf.train.latest_checkpoint(checkpoint_dir))
                test_eval = evaluate_model(model, test_batched_dataset, nb_class, args)
                (
                    epoch_micro_precision,
                    epoch_macro_precision,
                    epoch_micro_recall,
                    epoch_macro_recall,
                    epoch_micro_accuracy,
                    epoch_macro_accuracy,
                    epoch_micro_f1score,
                    epoch_macro_f1score,
                    epoch_micro_f2score,
                    epoch_macro_f2score,
                ) = test_eval
                if args["worker_index"] == 0:
                    print("\n\n\n\n Test Scores \n\n\n=============")

                    print(
                        f"Epoch {epoch:03d}: micro: accuracy: {epoch_micro_accuracy:.3f}, precision: {epoch_micro_precision:.3f}, recall: {epoch_micro_recall:.3f}, f1-score: {epoch_micro_f1score:.3f}, f2-score: {epoch_micro_f2score:.3f}"
                    )
                    print(
                        f"Epoch {epoch:03d}: macro: accuracy: {epoch_macro_accuracy:.3f}, precision: {epoch_macro_precision:.3f}, recall: {epoch_macro_recall:.3f}, f1-score: {epoch_macro_f1score:.3f}, f2-score: {epoch_macro_f2score:.3f}"
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument(
        "--configs",
        required=False,
        default="configs/base.json",
        help="JSON config file",
    )
    parser.add_argument(
        "--parallel", required=False, default=False, help="Enable parallelism"
    )
    parser_args = parser.parse_args()

    with open(parser_args.configs, "rb") as f:
        args = json.load(f)
        args.update(vars(parser_args))

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for d in gpus:
        tf.config.experimental.set_memory_growth(d, True)

    if parser_args.parallel:
        import horovod.tensorflow as hvd

        hvd.init()
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")
        args["num_workers"] = hvd.size()
        args["worker_index"] = hvd.rank()
    else:
        args["num_workers"] = 1
        args["worker_index"] = 0

    run_model(args)

    if parser_args.parallel:
        hvd.join()
