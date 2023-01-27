import torch.nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import tensorflow as tf
import numpy as np

from src.expred import ExpredInput
from src.fastif.influence_utils.nn_influence_utils import compute_hessian_vector_products, compute_gradients


def initialize_strategy():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
        print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
    except ValueError:
        raise BaseException(
            'ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

    return strategy


def get_models_lists(model, strategy):
    with strategy.scope():
        models_penultimate = []
        models_last = []
        for i in [30, 60, 90]:
            model = resnet.resnet50(1000)
            model.load_weights(CHECKPOINTS_PATH_FORMAT.format(i))
            models_penultimate.append(tf.keras.Model(model.layers[0].input, model.layers[-3].output))
            models_last.append(model.layers[-2])


def tracin_run(model: torch.nn.Module, inputs: ExpredInput, train: DataLoader):
    layer_filter = [
        n for n, p in model.named_parameters()
        if not p.requires_grad]

    model.cuda()

    weight_decay = 0.005
    num_samples = len(train)
    damp = 3e-5,
    scale = 1e4,

    weight_decay_ignores = [
        "bias",
        "LayerNorm.weight"]

    penultimate = None
    for _ in range(1):

        if layer_filter is None:
            layer_filter = []

        model.zero_grad()
        outputs = model(inputs)

        preds = outputs['cls_preds']
        loss = max(preds['cls_pred'][0][0], preds['cls_pred'][0][1])

        no_decay = (
            weight_decay_ignores
            if weight_decay_ignores
               is not None else [])

        weight_decay_loss = torch.cat([
            p.square().view(-1)
            for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ]).sum() * weight_decay

        loss = loss + weight_decay_loss

        cls_grad = torch.autograd.grad(
            outputs=loss,
            inputs=[
                param for name, param
                in model.cls_module.named_parameters()
                if name not in layer_filter],
            create_graph=True,
            retain_graph=True,
            allow_unused=True)

        grad = cls_grad

        last_checkpoint = list(grad).copy()
        cumulative_num_samples = 0
        with tqdm(total=num_samples) as pbar:
            for data_loader in [train]:
                for i, inputs in enumerate(data_loader):

                    model.zero_grad()
                    outputs = model(inputs[0])

                    preds = outputs['cls_preds']
                    loss = max(preds['cls_pred'][0][0], preds['cls_pred'][0][1])

                    if weight_decay is not None:
                        no_decay = (
                            weight_decay_ignores
                            if weight_decay_ignores
                               is not None else [])

                        weight_decay_loss = torch.cat([
                            p.square().view(-1)
                            for n, p in model.named_parameters()
                            if not any(nd in n for nd in no_decay)
                        ]).sum() * weight_decay
                        loss = loss + weight_decay_loss

                    grad_tuple = torch.autograd.grad(
                        outputs=loss,
                        inputs=[
                            param for name, param
                            in model.named_parameters()
                            if name not in layer_filter],
                        create_graph=True,
                        allow_unused=True
                    )

                    model.zero_grad()
                    this_checkpoint = torch.autograd.grad(
                        outputs=grad_tuple[0:2],
                        inputs=[
                            param for name, param
                            in model.named_parameters()
                            if name not in layer_filter],
                        grad_outputs=last_checkpoint[0:2],
                        only_inputs=True,
                        allow_unused=True
                    )

                    with torch.no_grad():
                        new_checkpoint = [
                            a + (1 - damp) * b - c / scale
                            for a, b, c in zip(grad[0:2], last_checkpoint[0:2], this_checkpoint[0:2])
                        ]

                    pbar.update(1)
                    new_checkpoint_norm = new_checkpoint[0].norm().item()
                    last_checkpoint_norm = last_checkpoint[0].norm().item()
                    checkpoint_norm_diff = new_checkpoint_norm - last_checkpoint_norm
                    pbar.set_description(f"{new_checkpoint_norm:.2f} | {checkpoint_norm_diff:.2f}")

                    cumulative_num_samples += 1
                    last_checkpoint = new_checkpoint
                    if num_samples is not None and i > num_samples:
                        break

        _penultimate = [X / scale for X in last_checkpoint]

        if penultimate is None:
            penultimate = _penultimate
        else:
            penultimate = [
                a + b for a, b in zip(penultimate, _penultimate)
            ]

    penultimate = [a / 1 for a in penultimate]

    influences = {}
    train_inputs_collections = {}
    for index, train_inputs in enumerate(tqdm(train)):

        ann_ids = train_inputs[1]
        train_inputs = train_inputs[0]

        if layer_filter is None:
            layer_filter = []

        model.zero_grad()
        outputs = model(inputs)

        preds = outputs['cls_preds']
        loss = max(preds['cls_pred'][0][0], preds['cls_pred'][0][1])

        if weight_decay is not None:
            no_decay = (
                weight_decay_ignores
                if weight_decay_ignores
                   is not None else [])

            weight_decay_loss = torch.cat([
                p.square().view(-1)
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ]).sum() * weight_decay
            loss = loss + weight_decay_loss

        grad_z = torch.autograd.grad(
            outputs=loss,
            inputs=[
                param for name, param
                in model.cls_module.named_parameters()
                if name not in layer_filter],
            create_graph=True,
            retain_graph=True,
            allow_unused=True)

        with torch.no_grad():
            influence = [
                - torch.sum(x * y)
                for x, y in zip(grad_z, penultimate)]

        influences[index] = (ann_ids, sum(influence).item())
        train_inputs_collections[index] = train_inputs

    return influences


def get_trackin_grad(model: torch.nn.Module, inputs: ExpredInput, train: DataLoader):
    strategy = initialize_strategy()
    annotation_ids_np = []
    loss_grads_np = []
    activations_np = []
    labels_np = []
    probs_np = []
    predicted_labels_np = []
    for i, d in enumerate(tqdm(train)):
        annotation_ids_replicas, loss_grads_replica, activations_replica, labels_replica, probs_replica, predictied_labels_replica = strategy.run(
            run, args=(model, inputs, train))
        for annotation_ids, loss_grads, activations, labels, probs, predicted_labels in zip(
                strategy.experimental_local_results(annotation_ids_replicas),
                strategy.experimental_local_results(loss_grads_replica),
                strategy.experimental_local_results(activations_replica),
                strategy.experimental_local_results(labels_replica),
                strategy.experimental_local_results(probs_replica),
                strategy.experimental_local_results(predictied_labels_replica)):
            if annotation_ids.shape[0] == 0:
                continue
            annotation_ids_np.append(annotation_ids.numpy())
            loss_grads_np.append(loss_grads.numpy())
            activations_np.append(activations.numpy())
            labels_np.append(labels.numpy())
            probs_np.append(probs.numpy())
            predicted_labels_np.append(predicted_labels.numpy())
    return {'annotation_ids': np.concatenate(annotation_ids_np),
            'loss_grads': np.concatenate(loss_grads_np),
            'activations': np.concatenate(activations_np),
            'labels': np.concatenate(labels_np),
            'probs': np.concatenate(probs_np),
            'predicted_labels': np.concatenate(predicted_labels_np)
            }


def find(loss_grad=None, activation=None, topk=50):
    if loss_grad is None and activation is None:
        raise ValueError('loss grad and activation cannot both be None.')
    scores = []
    scores_lg = []
    scores_a = []
    for i in range(len(trackin_train['image_ids'])):
        if loss_grad is not None and activation is not None:
            lg_sim = np.sum(trackin_train['loss_grads'][i] * loss_grad, axis=0)
            a_sim = np.sum(trackin_train['activations'][i] * activation, axis=0)
            scores.append(np.sum(lg_sim * a_sim))
            scores_lg.append(np.sum(lg_sim))
            scores_a.append(np.sum(a_sim))
        elif loss_grad is not None:
            scores.append(np.sum(trackin_train['loss_grads'][i] * loss_grad))
        elif activation is not None:
            scores.append(np.sum(trackin_train['activations'][i] * activation))

    opponents = []
    proponents = []
    indices = np.argsort(scores)
    for i in range(topk):
        index = indices[-i - 1]
        proponents.append((
            trackin_train['image_ids'][index],
            trackin_train['probs'][index][0],
            index_to_classname[str(trackin_train['predicted_labels'][index][0])][1],
            index_to_classname[str(trackin_train['labels'][index])][1],
            scores[index],
            scores_lg[index] if scores_lg else None,
            scores_a[index] if scores_a else None))
        index = indices[i]
        opponents.append((
            trackin_train['image_ids'][index],
            trackin_train['probs'][index][0],
            index_to_classname[str(trackin_train['predicted_labels'][index][0])][1],
            index_to_classname[str(trackin_train['labels'][index])][1],
            scores[index],
            scores_lg[index] if scores_lg else None,
            scores_a[index] if scores_a else None))
    return opponents, proponents