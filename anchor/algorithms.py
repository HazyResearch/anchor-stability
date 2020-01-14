import logging
import os
import subprocess

from anchor.embedding import DualEmbedding, Embedding

logger = logging.getLogger(__name__)


class Algorithm:
    def _call(self, cmd, logfile):
        f = open(logfile, "w")
        logger.debug(f"Running: {cmd}")
        # propagate any errors up to python
        subprocess.check_call(cmd, stdout=f, stderr=f, shell=True)

    def run(self, working_dir, **kwargs):
        raise NotImplementedError

    def run_warmstart(self, working_dir, warmstart, **kwargs):
        raise NotImplementedError


class Word2Vec(Algorithm):
    def __init__(
        self,
        exec_dir,
        tag,
        dim=50,
        window=15,
        negative=5,
        threads=56,
        epoch=100,
        min_count=5,
        seed=1234,
        checkpoint_interval=100,
        cbow=False,
        **kwargs,
    ):
        self.exec_dir = exec_dir
        self.tag = tag
        self.size = dim
        self.window = window
        self.negative = negative
        self.threads = threads
        self.epoch = epoch
        self.min_count = min_count
        self.seed = seed
        self.checkpoint_interval = checkpoint_interval
        if "alpha" not in kwargs:
            if cbow:
                self.alpha = 0.05
            else:
                self.alpha = 0.025
        else:
            self.alpha = kwargs.get("alpha")
        self.cbow = cbow
        self.header = True

    def run(self, working_dir, data, tag=None, vocab=None, **kwargs):
        if tag is None:
            full_tag = f"{working_dir}/{self.tag}"
        else:
            full_tag = f"{working_dir}/{tag}"
        if self.cbow:
            cmd = f"{self.exec_dir} -alpha {self.alpha} -train {data} -output {full_tag} -cbow 1 -size {self.size} -binary 0 -window {self.window} -negative {self.negative} -threads {self.threads} -iter {self.epoch} -min-count {self.min_count} -seed {self.seed} -checkpoint_interval {self.checkpoint_interval}"
        else:
            cmd = f"{self.exec_dir} -alpha {self.alpha} -train {data} -output {full_tag} -cbow 0 -size {self.size} -binary 0 -window {self.window} -negative {self.negative} -threads {self.threads} -iter {self.epoch} -min-count {self.min_count} -seed {self.seed} -checkpoint_interval {self.checkpoint_interval}"
        if vocab is not None:
            cmd += f" -read-vocab {vocab}"
        logfile = f"{full_tag}.log"
        self._call(cmd, logfile)
        # Save paths to word and context embeddings
        word_path = f"{full_tag}.{self.epoch}.w.txt"
        context_path = f"{full_tag}.{self.epoch}.c.txt"
        return DualEmbedding(
            word_path=word_path, context_path=context_path, header=self.header
        )

    def run_warmstart(
        self, working_dir, warmstart, data, tag=None, vocab=None, **kwargs
    ):
        if tag is None:
            full_tag = f"{working_dir}/{self.tag}"
        else:
            full_tag = f"{working_dir}/{tag}"
        init_word = warmstart.w.emb_path
        init_context = warmstart.c.emb_path
        if self.cbow:
            cmd = f"{self.exec_dir} -alpha {self.alpha} -train {data} -output {full_tag} -cbow 1 -size {self.size} -binary 0 -window {self.window} -negative {self.negative} -threads {self.threads} -iter {self.epoch} -min-count {self.min_count} -seed {self.seed} -checkpoint_interval {self.checkpoint_interval} -init-word {init_word} -init-context {init_context}"
        else:
            cmd = f"{self.exec_dir} -alpha {self.alpha} -train {data} -output {full_tag} -cbow 0 -size {self.size} -binary 0 -window {self.window} -negative {self.negative} -threads {self.threads} -iter {self.epoch} -min-count {self.min_count} -seed {self.seed} -checkpoint_interval {self.checkpoint_interval} -init-word {init_word} -init-context {init_context}"
        if vocab is not None:
            cmd += f" -read-vocab {vocab}"
        logfile = f"{full_tag}.log"
        self._call(cmd, logfile)
        # Save paths to word and context embeddings
        word_path = f"{full_tag}.{self.epoch}.w.txt"
        context_path = f"{full_tag}.{self.epoch}.c.txt"
        return DualEmbedding(
            word_path=word_path, context_path=context_path, header=self.header
        )


class HazyEmbeddings(Algorithm):
    def _prepare_data(self, data, vocab, coo, working_dir):
        # user needs to pass data if not giving vocab or coo
        if vocab is None or coo is None:
            assert data is not None

        if vocab is None:
            # generate vocab
            vocab = f"{working_dir}/vocab_{os.path.basename(data)}.txt"
            cmd = f"{self.exec_dir}/vocab_count -min-count {self.min_count} -data-file {data} -vocab-file {vocab}"
            logfile = f"{working_dir}/vocab_{os.path.basename(data)}.log"
            self._call(cmd, logfile)

        if coo is None:
            coo = f"{working_dir}/coo_{os.path.basename(data)}.bin"
            cmd = f"{self.exec_dir}/cooccur -window-size {self.window_size} -vocab-file {vocab} -memory {self.memory} -data-file {data} -output {coo}"
            logfile = f"{working_dir}/coo_{os.path.basename(data)}.log"
            self._call(cmd, logfile)

        return data, vocab, coo

    def run(self, working_dir, **kwargs):
        raise NotImplementedError

    def run_warmstart(self, working_dir, warmstart, **kwargs):
        raise NotImplementedError


class PowerIteration(HazyEmbeddings):
    def __init__(
        self,
        exec_dir,
        tag,
        dim=50,
        threads=56,
        epoch=100,
        checkpoint_interval=100,
        log_interval=100,
        seed=1234,
        tol=1e-4,
        min_count=5,
        window_size=15,
        memory=500,
    ):
        self.exec_dir = exec_dir
        self.tag = tag
        self.dim = dim
        self.threads = threads
        self.epoch = epoch
        self.checkpoint_interval = checkpoint_interval
        self.log_interval = log_interval
        self.seed = seed
        self.tol = tol
        self.min_count = min_count
        self.window_size = window_size
        self.memory = memory

    def run(self, working_dir, data=None, vocab=None, coo=None, tag=None, **kwargs):
        data, vocab, coo = self._prepare_data(
            data=data, vocab=vocab, coo=coo, working_dir=working_dir
        )
        if tag is None:
            full_tag = f"{working_dir}/{self.tag}"
        else:
            full_tag = f"{working_dir}/{tag}"
        cmd = f"{self.exec_dir}/embedding -x pi -f {coo} -v {vocab} -i {self.epoch} -d {self.dim} -t {self.threads} -o {full_tag} -s {self.seed} -w {self.seed} -z {self.checkpoint_interval} -p {self.log_interval} -e {self.tol}"
        logfile = f"{full_tag}.log"
        self._call(cmd, logfile)
        emb_path = f"{full_tag}.{self.epoch}.final"
        return Embedding(emb_path)

    def run_warmstart(
        self,
        working_dir,
        warmstart,
        data=None,
        vocab=None,
        coo=None,
        tag=None,
        **kwargs,
    ):
        data, vocab, coo = self._prepare_data(
            data=data, vocab=vocab, coo=coo, working_dir=working_dir
        )
        if tag is None:
            full_tag = f"{working_dir}/{self.tag}"
        else:
            full_tag = f"{working_dir}/{tag}"
        cmd = f"{self.exec_dir}/embedding -x pi -f {coo} -v {vocab} -i {self.epoch} -d {self.dim} -t {self.threads} -o {full_tag} -s {self.seed} -w {self.seed} -z {self.checkpoint_interval} -p {self.log_interval} -e {self.tol} -r {warmstart.emb_path} -n 1"
        logfile = f"{full_tag}.log"
        self._call(cmd, logfile)
        emb_path = f"{full_tag}.{self.epoch}.final"
        return Embedding(emb_path)


class MatrixCompletion(HazyEmbeddings):
    def __init__(
        self,
        exec_dir,
        tag,
        dim=50,
        threads=56,
        epoch=100,
        checkpoint_interval=100,
        log_interval=100,
        seed=1234,
        tol=1e-4,
        min_count=5,
        window_size=15,
        memory=500,
        lr=1,
        beta=0,
        reg=0,
        lr_decay=20,
        batch_size=128,
    ):
        self.exec_dir = exec_dir
        self.tag = tag
        self.dim = dim
        self.threads = threads
        self.epoch = epoch
        self.checkpoint_interval = checkpoint_interval
        self.log_interval = log_interval
        self.seed = seed
        self.tol = tol
        self.min_count = min_count
        self.window_size = window_size
        self.memory = memory
        self.lr = lr
        self.beta = beta
        self.reg = reg
        self.lr_decay = lr_decay
        self.batch_size = batch_size

    def run(self, working_dir, data=None, vocab=None, coo=None, tag=None, **kwargs):
        data, vocab, coo = self._prepare_data(
            data=data, vocab=vocab, coo=coo, working_dir=working_dir
        )
        if tag is None:
            full_tag = f"{working_dir}/{self.tag}"
        else:
            full_tag = f"{working_dir}/{tag}"
        cmd = f"{self.exec_dir}/embedding -x sgd -f {coo} -v {vocab} -i {self.epoch} -d {self.dim} -t {self.threads} -o {full_tag} -s {self.seed} -w {self.seed} -z {self.checkpoint_interval} -p {self.log_interval} -e {self.tol} -b {self.batch_size} -l {self.lr} -a {self.beta} -y {self.lr_decay} -m {self.reg}"
        logfile = f"{full_tag}.log"
        self._call(cmd, logfile)
        emb_path = f"{full_tag}.{self.epoch}.final"
        return Embedding(emb_path)

    def run_warmstart(
        self,
        working_dir,
        warmstart,
        data=None,
        vocab=None,
        coo=None,
        tag=None,
        **kwargs,
    ):
        data, vocab, coo = self._prepare_data(
            data=data, vocab=vocab, coo=coo, working_dir=working_dir
        )
        if tag is None:
            full_tag = f"{working_dir}/{self.tag}"
        else:
            full_tag = f"{working_dir}/{tag}"
        cmd = f"{self.exec_dir}/embedding -x sgd -f {coo} -v {vocab} -i {self.epoch} -d {self.dim} -t {self.threads} -o {full_tag} -s {self.seed} -w {self.seed} -z {self.checkpoint_interval} -p {self.log_interval} -e {self.tol} -b {self.batch_size} -l {self.lr} -a {self.beta} -y {self.lr_decay} -m {self.reg} -r {warmstart.emb_path} -n 1"
        logfile = f"{full_tag}.log"
        self._call(cmd, logfile)
        emb_path = f"{full_tag}.{self.epoch}.final"
        return Embedding(emb_path)


class GloVe(Algorithm):
    def __init__(
        self,
        exec_dir,
        tag,
        dim=50,
        threads=56,
        alpha=0.75,
        x_max=100.0,
        eta=0.05,
        seed=1234,
        checkpoint_interval=100,
        epoch=100,
        min_count=5,
        window_size=15,
        memory=500,
    ):
        self.exec_dir = exec_dir
        self.tag = tag
        self.size = dim
        self.threads = threads
        self.alpha = alpha
        self.x_max = x_max
        self.eta = eta
        self.seed = seed
        self.checkpoint_interval = checkpoint_interval
        self.epoch = epoch
        self.min_count = min_count
        self.window_size = window_size
        self.memory = memory
        self.header = True

    def _prepare_data(self, data, vocab, coo, working_dir):
        # user needs to pass data if not giving vocab or coo
        if vocab is None or coo is None:
            assert data is not None
        if vocab is None:
            # generate vocab
            logger.info("Generating vocabulary file")
            vocab = f"{working_dir}/vocab_{os.path.basename(data)}.txt"
            cmd = f"{self.exec_dir}/vocab_count -min-count {self.min_count} < {data} > {vocab}"
            logfile = f"{working_dir}/vocab_{os.path.basename(data)}.log"
            self._call(cmd, logfile)
        if coo is None:
            logger.info("Generating coo")
            coo = f"{working_dir}/coo_{os.path.basename(data)}.bin"
            cmd = f"{self.exec_dir}/cooccur -window-size {self.window_size} -vocab-file {vocab} -memory {self.memory} < {data} > {coo}"
            logfile = f"{working_dir}/coo_{os.path.basename(data)}.log"
            self._call(cmd, logfile)

            # shuffle coo
            coo_shuf = f"{working_dir}/coo_{os.path.basename(data)}.shuf.bin"
            cmd = f"{self.exec_dir}/shuffle -memory {self.memory} < {coo} > {coo_shuf}"
            logfile = f"{working_dir}/coo_shuf_{os.path.basename(data)}.log"
            self._call(cmd, logfile)
            coo = coo_shuf
        return data, vocab, coo

    def run(self, working_dir, data=None, vocab=None, coo=None, tag=None, **kwargs):
        data, vocab, coo = self._prepare_data(
            data=data, vocab=vocab, coo=coo, working_dir=working_dir
        )
        if tag is None:
            full_tag = f"{working_dir}/{self.tag}"
        else:
            full_tag = f"{working_dir}/{tag}"
        cmd = f"{self.exec_dir}/glove -save-file {full_tag} -input-file {coo} -model 2 -vocab-file {vocab} -vector-size {self.size} -binary 0 -threads {self.threads} -seed {self.seed} -checkpoint-every {self.checkpoint_interval} -iter {self.epoch} -eta {self.eta}"
        logfile = f"{full_tag}.log"
        self._call(cmd, logfile)
        word_path = f"{full_tag}.w.txt"
        context_path = f"{full_tag}.c.txt"
        return DualEmbedding(
            word_path=word_path, context_path=context_path, header=self.header
        )

    def run_warmstart(
        self,
        working_dir,
        warmstart,
        data=None,
        vocab=None,
        coo=None,
        tag=None,
        **kwargs,
    ):
        data, vocab, coo = self._prepare_data(
            data=data, vocab=vocab, coo=coo, working_dir=working_dir
        )
        if tag is None:
            full_tag = f"{working_dir}/{self.tag}"
        else:
            full_tag = f"{working_dir}/{tag}"
        init_word = warmstart.w.emb_path
        init_context = warmstart.c.emb_path
        cmd = f"{self.exec_dir}/glove -save-file {full_tag} -input-file {coo} -model 2 -vocab-file {vocab} -vector-size {self.size} -binary 0 -threads {self.threads} -seed {self.seed} -checkpoint-every {self.checkpoint_interval} -iter {self.epoch} -init-word {init_word} -init-context {init_context} -eta {self.eta}"
        logfile = f"{full_tag}.log"
        self._call(cmd, logfile)
        word_path = f"{full_tag}.w.txt"
        context_path = f"{full_tag}.c.txt"
        return DualEmbedding(
            word_path=word_path, context_path=context_path, header=self.header
        )
