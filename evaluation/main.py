import json
import math
import numpy as np
import pickle as pkl
import os
import time
import sys
import traceback

from time import sleep
from threading import Thread, Event

import torch
from torch.utils.data import RandomSampler

from nas import NAS
from data_processor import DataProcessor
from trainer import Trainer

#Time limit in hours
TIME_LIMIT = 12

# === DATA LOADING HELPERS =============================================================================================
# find the dataset filepaths
def get_dataset_paths(data_dir):
    paths = sorted([os.path.join(data_dir, d) for d in os.listdir(data_dir) if 'dataset' in d], reverse=True)
    return paths

# load the dataset metadata from json
def load_dataset_metadata(dataset_path):
    with open(os.path.join(dataset_path, 'metadata'), "r") as f:
        metadata = json.load(f)
    return metadata

# load dataset from file
def load_datasets(data_path, truncate):
    data_path = 'datasets/'+data_path
    train_x = np.load(os.path.join(data_path,'train_x.npy'))
    train_y = np.load(os.path.join(data_path,'train_y.npy'))
    valid_x = np.load(os.path.join(data_path,'valid_x.npy'))
    valid_y = np.load(os.path.join(data_path,'valid_y.npy'))
    test_x = np.load(os.path.join(data_path,'test_x.npy'))
    metadata = load_dataset_metadata(data_path)

    if truncate:
        train_x = train_x[:64]
        train_y = train_y[:64]
        valid_x = valid_x[:64]
        valid_y = valid_y[:64]
        test_x = test_x[:64]

    return (train_x, train_y), \
           (valid_x, valid_y), \
           (test_x), metadata


# === TIME COUNTERs ====================================================================================================
def div_remainder(n, interval):
    # finds divisor and remainder given some n/interval
    factor = math.floor(n / interval)
    remainder = int(n - (factor * interval))
    return factor, remainder


def show_time(seconds):
    # show amount of time as human readable
    if seconds < 60:
        return "{:.2f}s".format(seconds)
    elif seconds < (60 * 60):
        minutes, seconds = div_remainder(seconds, 60)
        return "{}m,{}s".format(minutes, seconds)
    else:
        hours, seconds = div_remainder(seconds, 60 * 60)
        minutes, seconds = div_remainder(seconds, 60)
        return "{}h,{}m,{}s".format(hours, minutes, seconds)


# keep a counter of available time
class Clock:
    def __init__(self, time_limit_in_hours):
        self.start_time =  time.perf_counter()
        self.time_limit = self.start_time + (time_limit_in_hours * 60 * 60)

    def check(self):
        return (self.time_limit - time.perf_counter())


def countdown(e:Event, time_limit:int):
    counter = 0
    while time.perf_counter() < time_limit:
        counter += 1
        if counter == 10:
            print(f'Time Remaining: {show_time(time_limit - time.perf_counter())}')
            counter = 0
        sleep(1)
    print("Submission exceeded time_limit")
    e.set()


# === MODEL ANALYSIS ===================================================================================================
def general_num_params(model):
    # return number of differential parameters of input model
    return sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())])


# === MAIN =============================================================================================================
# the available runtime will change at various stages of the competition, but feel free to change for local tests
# note, this is approximate, your runtime will be controlled externally by our server
def main():
    # print main header
    print("=" * 78)
    print("="*13 + "    Your NAS Unseen-Data 2025 Submission is running     " + "="*13)
    print("="*78)

    # start tracking submission runtime
    runclock = Clock(TIME_LIMIT)

    e = Event()

    t1 = Thread(target=run_wrapper, args=[e, runclock])
    t2 = Thread(target=countdown, args=[e, runclock.time_limit])

    t1.daemon = True
    t2.daemon = True

    t1.start()
    t2.start()

    e.wait()
    return

def run_wrapper(e:Event, runclock:Clock):
    try:
        run_submission(runclock)
    except Exception as e:
        print(e)
        print(traceback.format_exc())
    e.set()


def run_submission(runclock:Clock):
        # iterate over datasets in the datasets directory
        for dataset in os.listdir("datasets"):
            # load and display data info
            (train_x, train_y), (valid_x, valid_y), (test_x), metadata = load_datasets(dataset, truncate=False)
            metadata['time_remaining'] = runclock.check()
            this_dataset_start_time = time.perf_counter()

            print("="*10 + " Dataset {:^10} ".format(metadata['codename']) + "="*45)
            print("  Metadata:")
            [print("   - {:<20}: {}".format(k, v)) for k,v in metadata.items()]

            # perform data processing/augmentation/etc using your DataProcessor
            print("\n=== Processing Data ===")
            print("  Allotted compute time remaining: ~{}".format(show_time(runclock.check())))
            data_processor = DataProcessor(train_x, train_y, valid_x, valid_y, test_x, metadata, runclock)
            train_loader, valid_loader, test_loader = data_processor.process()
            metadata['time_remaining'] = runclock.check()

            # check that the test_loader is configured correctly
            assert_string = "Test Dataloader is {}, this will break evaluation. Please fix this in your DataProcessor init."
            assert not isinstance(test_loader.sampler, RandomSampler), assert_string.format("shuffling")
            assert not test_loader.drop_last, assert_string.format("dropping last batch")

            # search for best model using your NAS algorithm
            print("\n=== Performing NAS ===")
            print("  Allotted compute time remaining: ~{}".format(show_time(runclock.check())))
            model = NAS(train_loader, valid_loader, metadata, runclock).search()
            model_params = int(general_num_params(model))
            metadata['time_remaining'] = runclock.check()

            # train model using your Trainer
            print("\n=== Training ===")
            print("  Allotted compute time remaining: ~{}".format(show_time(runclock.check())))
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
            trainer = Trainer(model, device, train_loader, valid_loader, metadata, runclock)
            trained_model = trainer.train()

            # submit predictions to file
            print("\n=== Predicting ===")
            print("  Allotted compute time remaining: ~{}".format(show_time(runclock.check())))
            predictions = trainer.predict(test_loader)
            run_data = {'Runtime': float(np.round(time.perf_counter()-this_dataset_start_time, 2)), 'Params': model_params}
            with open("predictions/{}_stats.pkl".format(metadata['codename']), "wb") as f:
                pkl.dump(run_data, f)
            np.save('predictions/{}.npy'.format(metadata['codename']), predictions)
            print("Model Training and Prediction Complete")

if __name__ == '__main__':
    main()