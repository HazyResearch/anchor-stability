import logging
import os
import subprocess
from enum import Enum

import coloredlogs

import anchor.utils as utils
from anchor.embedding import DualEmbedding, Embedding

coloredlogs.install(level="DEBUG")
logger = logging.getLogger(__name__)


class Initializer(Enum):
    RANDOM = 0
    WARMSTART = 1
    OPTIMIZER_ONE_EPOCH = 2
    OPTIMIZER_SAMPLE = 3


class Ensemble(Enum):
    ALL = 0
    NONE = 1
    OPTIMIZER = 2


class Anchor:
    def __init__(
        self,
        emb_dir=".",
        initializer=Initializer.RANDOM,
        ensemble=Ensemble.NONE,
        previous_embs=[],
        reference=None,
    ):
        self.reference = reference
        self.previous_embs = []
        # TODO(mleszczy): Only accept embedding objects?
        if len(previous_embs) > 0:
            if not isinstance(previous_embs, list):
                previous_embs = [previous_embs]
            for emb in previous_embs:
                # if isinstance(emb, str):
                #     self.previous_embs.append(Embedding(emb))
                # elif isinstance(emb, tuple):
                #     self.previous_embs.append(DualEmbedding(emb[0], emb[1]))
                if isinstance(emb, Embedding) or isinstance(emb, DualEmbedding):
                    self.previous_embs.append(emb)
                    if emb.reference is None:
                        emb.reference = emb
                    if self.reference is not None:
                        assert (
                            self.reference == emb.reference
                        ), "Embeddings with different references not currently supported"
                    else:
                        self.reference = emb.reference
                        logging.info(f"Setting reference to {emb.reference}")
                else:
                    raise ValueError(
                        "Invalid form for previous embeddings. Must be of type Embedding, or DualEmbedding."
                    )
        self.emb_dir = emb_dir
        os.makedirs(emb_dir, exist_ok=True)
        self.initializer = initializer
        self.ensemble = ensemble

    def reset(self):
        self.previous_embs = []

    def gen_embedding(self, algo, **kwargs):
        # set up logging
        log_file = kwargs.get("log_file", "system.log")
        utils.create_logger(log_file)
        git_hash = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .strip()
            .decode("utf-8")
        )
        logging.debug(f"Git hash: {git_hash}")

        # if user didn't specify for this training period, use class initializer
        if "initializer" not in kwargs:
            initializer = self.initializer

        # generate the new embedding
        new_emb = self._train(algo=algo, initializer=initializer, **kwargs)

        # save aligned embedding
        self._align_embedding(new_emb=new_emb)

        # ensemble with previous embeddings to materialize to user
        new_emb = self._ensemble_embeddings()
        return new_emb

    def _train(self, algo, initializer, **kwargs):
        """ Trains a new embedding with algo using data and initializer """
        logger.info(f"Using initializer: {initializer}")

        if len(self.previous_embs) == 0 or self.initializer is Initializer.RANDOM:
            warmstart = None

        # do training
        elif self.initializer is Initializer.WARMSTART:
            warmstart = self.previous_embs[-1]

        if warmstart is not None:
            logger.info("Using warmstart")
            new_emb = algo.run_warmstart(
                working_dir=self.emb_dir, warmstart=warmstart, **kwargs
            )
        else:
            new_emb = algo.run(working_dir=self.emb_dir, **kwargs)

        # store embedding objects
        self.previous_embs.append(new_emb)
        return new_emb

    def _align_embedding(self, new_emb):
        # if first embedding then embedding is the reference
        if self.reference is None:
            self.reference = new_emb

        # don't need to resave it's aligned to itself
        else:
            new_emb.align(self.reference)

            # write new embedding, replacing old embedding
            new_emb.save()

    def _ensemble_embeddings(self):
        """ Ensembles embeddings. Currently only ensembles words which appear across all embeddings in the ensemble. """

        if len(self.previous_embs) == 1 or self.ensemble is Ensemble.NONE:
            return self.previous_embs[0]

        # Only ensemble up to threshold
        elif self.ensemble is Ensemble.OPTIMIZER:
            # Use optimizer to ensemble embeddings
            pass

        elif self.ensemble is Ensemble.ALL:
            pass

        # TODO(mleszczy): Combine ensemble with new words?
        cls_ = type(self.previous_embs[0])
        logger.info(f"Embeddings stored with class {cls_}")
        return cls_.ensemble_embeddings(embeddings=self.previous_embs)
