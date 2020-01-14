import os
import subprocess

def check_ner_complete(modelpath1, modelpath2):
    """ Checks that the NER task is complete by locating the final-model artifact. """
    file1_final = f"{modelpath1}/final-model.pt"
    file2_final = f"{modelpath2}/final-model.pt"
    assert os.path.isfile(file1_final), file1_final
    assert os.path.isfile(file2_final), file2_final

def check_sent_complete(modelpath1, modelpath2):
    """Checks that the logs for the sentence analysis task are complete."""
    try:
        ff = open(f'{modelpath1}.log', 'r')
        dat = [_.strip() for _ in ff]
        error1 = 1-float(dat[-2].strip().split(': ')[1])
    except:
        return False

    try:
        ff = open(f'{modelpath2}.log', 'r')
        dat = [_.strip() for _ in ff]
        error2  = 1-float(dat[-2].strip().split(': ')[1])
    except:
        return False

    return True

def run_task(taskdir, taskfile, embpath):
    results = subprocess.check_output(
        ["python", os.path.join(taskdir, "ws_eval.py"), "GLOVE", embpath, taskfile]
    )
    correlation = results.decode("utf8").strip().split(" ")[-1]
    return float(correlation)