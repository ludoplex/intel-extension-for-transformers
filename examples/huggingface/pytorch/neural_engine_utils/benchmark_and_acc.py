"""This script is for NLP Neural_Engine benchmaking."""
import os
import subprocess
import re
import argparse
import shutil

def utils_bak(path):
    """backup utils.py."""
    src = f"{path}/utils.py"
    src_bak = f"{path}/utils.py.bak"
    shutil.copyfile(src, src_bak)

def utils_copy_after_benchmark(path):
    """copy backup file to original."""
    src = f"{path}/utils.py.bak"
    src_bak = f"{path}/utils.py"
    shutil.copyfile(src, src_bak)



def modify_sequence(path, seq_len, dataset_reorder):
    """change sequence len and dataset order."""
    utils_bak(path)
    src = f"{path}/utils.py"
    src_tmp = f"{path}/utils_tmp.py"
    with open(src, "r") as src_fp:
        with open(src_tmp, "w") as dst_fp:
            for line in src_fp:
                line_replace = line
                if line.find("np.array(segment_ids_data)") >= 0 and dataset_reorder == 1:
                    line_replace = line.replace("segment_ids_data", "input_mask_data")
                elif line.find("np.array(input_mask_data)") >= 0 and dataset_reorder == 1:
                    line_replace = line.replace("input_mask_data", "segment_ids_data")
                elif line.find("max_length=128") >= 0 and seq_len > 0:
                    line_replace = line.replace("max_length=128", f"max_length={seq_len}")
                elif line.find("F_max_seq_length = 128") >= 0 and seq_len > 0:
                    line_replace = line.replace(
                        "F_max_seq_length = 128",
                        f"F_max_seq_length = {seq_len}",
                    )
                elif line.find("F_max_seq_length = 384") >= 0 and seq_len > 0:
                    line_replace = line.replace(
                        "F_max_seq_length = 384",
                        f"F_max_seq_length = {seq_len}",
                    )
                dst_fp.write(line_replace)

    dst_fp.close()
    src_fp.close()
    shutil.copyfile(src_tmp, src)

def modify_yaml(
        path,
        framework,
        instance,
        cores,
        warmup,
        iteration,
        label_file,
        vocab_file):
    """we copy bert.yaml and change attribute."""
    with open(f"{path}/bert_static.yaml", "r") as src_fp:
        with open(f"{path}/bert_tmp.yaml", "w") as dst_fp:
            for line in src_fp:
                if line.find("num_of_instance") >= 0:
                    dst_fp.write(f"      num_of_instance: {instance}\n")
                elif line.find("cores_per_instance") >= 0:
                    dst_fp.write(f"      cores_per_instance: {cores}\n")
                elif line.find("warmup") >= 0:
                    dst_fp.write(f"    warmup: {warmup}\n")
                elif line.find("iteration") >= 0:
                    dst_fp.write(f"    iteration: {iteration}\n")
                elif line.find("label_file") >= 0:
                    dst_fp.write(f"          label_file: {label_file}\n")
                elif line.find("vocab_file") >= 0:
                    dst_fp.write(f"          vocab_file: {vocab_file}\n")
                elif line.find("framework") >= 0:
                    dst_fp.write(f"  framework: {framework}\n")
                else:
                    dst_fp.write(line)

    dst_fp.close()
    src_fp.close()


def numbers_to_strings(argument):
    """allocator mode num to str."""
    switcher = {
        0: "direct",
        1: "cycle",
        2: "unified",
        3: "je_direct",
        4: "je_cycle",
        5: "je_unified",
    }
    return switcher.get(argument, "cycle")


def concat_allocator_cmd(allocator, cmd):
    """add env variable for different allocator modes."""
    new_cmd = cmd
    if allocator == "direct":
        new_cmd = f"DIRECT_BUFFER=1 {cmd}"
    elif allocator == "unified":
        new_cmd = f"UNIFIED_BUFFER=1 {cmd}"
    elif allocator == "je_direct":
        new_cmd = f"JEMALLOC=1 DIRECT_BUFFER=1 {cmd}"
    elif allocator == "je_cycle":
        new_cmd = f"JEMALLOC=1 {cmd}"
    elif allocator == "je_unified":
        new_cmd = f"JEMALLOC=1 UNIFIED_BUFFER=1 {cmd}"
    return new_cmd


def grab_log(is_performance, path, instance, cores, log_fp):
    """extract performance from logs."""
    latency = float(0)
    throughput = float(0)
    if is_performance:
        i = 0
        throughput_str = ""
        latency_str = ""
        while i < instance:
            log_path = f"{path}/{instance}_{cores}_{i}.log"
            i += 1
            try:
                with open(log_path, 'r') as src_fp:
                    for line in src_fp:
                        if line.find("Throughput") >= 0:
                            throughput_str = line
                        elif line.find("Latency") >= 0:
                            latency_str = line
                float_re = re.compile(r'\d+\.\d+')
                floats = [float(i) for i in float_re.findall(throughput_str)]
                floats_latency = [float(i) for i in float_re.findall(latency_str)]

                throughput += floats[0]
                latency += floats_latency[0]
            except OSError as ex:
                print(ex)
        latency = latency / instance
    else:
        print("========please check acc with screen messages=============")
    try:
        if is_performance:
            log_fp.write(f"Troughput: {throughput} images/sec\n")
            log_fp.write(f"Latency: {latency} ms\n")
        log_fp.write("--------------------------------------\n")
    except OSError as ex:
        print(ex)


def execute_and_grab(is_performance, model_file, model_path, batch, allocator):
    """execute the run_engine.py."""
    cmd = ""
    if is_performance:
        cmd = f"GLOG_minloglevel=2 python run_engine.py --input_model={model_path}/{model_file} --config={model_path}/bert_tmp.yaml --benchmark --mode=performance --batch_size={batch}"
    else:

        cmd = f"GLOG_minloglevel=2 ONEDNN_VERBOSE=1 python run_engine.py --input_model={model_path}/{model_file} --config={model_path}/bert_tmp.yaml --benchmark --mode=accuracy --batch_size={batch}"

    cmd = concat_allocator_cmd(allocator, cmd)

    try:
        with open("tmp.sh", "w") as file_p:
            file_p.write(f"cd {model_path}\n")
            file_p.write(cmd)
        pro = subprocess.Popen(
            "bash tmp.sh",
            shell=True)
        pro.wait()
        file_p.close()

    except OSError as ex:
        print(ex)


def test_all(
        is_performance=True,
        support_models=None,
        batch=None,
        instance_cores=None,
        allocator_mode=None,
        sequence=128,
        warmup=5,
        iterations=10,
        is_int8=False,
        label_file="",
        vocab_file="",
        output_file=""):
    """find model and do benchmark."""
    print("search start")
    print(f"performance mode is {is_performance}")
    print(f"search for int8 model {is_int8}")
    benchmark_models = []
    benchmark_path = []
    if allocator_mode is None:
        allocator_mode = [1]
    if batch is None:
        batch = [1]
    if instance_cores is None:
        instance_cores = [1, 28]
    if support_models is None:
        support_models = ["bert_mini_mrpc"]


    for task in os.listdir(os.getcwd()):
        task_path = os.path.join(os.getcwd(), task)
        if os.path.isdir(task_path):
            for model in os.listdir(task_path):
                model_path = os.path.join(task_path, model)
                model_file = ""
                if not is_int8:
                    for file in os.listdir(model_path):
                        if file.endswith("onnx") or file.endswith("pb"):
                            model_file = file
                            benchmark_models = (*benchmark_models, model_file)
                            benchmark_path = (*benchmark_path, model_path)
                            print(model_file, " fp32 exist!!")
                            break
                else:
                    int8_model_path = os.path.join(model_path, "ir")
                    if os.path.exists(int8_model_path):
                        for file in os.listdir(int8_model_path):
                            if file.endswith("model.bin"):
                                model_file = file
                                benchmark_models = (*benchmark_models, "ir")
                                benchmark_path = (*benchmark_path, model_path)
                                break

                if model_file == "":
                    print(f"{model}_{task} not find model file!!!")
                elif f"{model}_{task}" not in support_models:
                    last_element_index = len(benchmark_models)-1
                    benchmark_models = benchmark_models[: last_element_index]
                    last_element_index = len(benchmark_path)-1
                    benchmark_path = benchmark_path[:last_element_index]
    print("search end")
    if not benchmark_models:
        print("============no .onnx or .pb for fp32, no ir folder for int8==============\n")
        return 0
    allocator = []

    instance = []
    cores = []
    instance, cores = zip(*instance_cores)
    dataset_reorder = 0
    framework = "engine"
    # this reorder and framework change only support for onnx model
    # tf model you need to use fp32 ir, so you should remvove snippet here
    # when model is tf, but we will not add arg to control, only bert base
    # and bert large use tf now
    if not is_int8 and not is_performance:
        dataset_reorder = 1
        framework = "onnxrt_integerops"
    print("============benchmark start==================")

    try:
        with open(output_file, "w") as file_p:
            for enabled_model_id, enabled_model_val in enumerate(
                    benchmark_models):
                print(enabled_model_val, "exist!!")
                bench_model_path = benchmark_path[enabled_model_id]
                bench_model_file = enabled_model_val
                modify_sequence(bench_model_path, sequence, dataset_reorder)
                for alloc_mode_id, alloc_mode_val in enumerate(allocator_mode):
                    allocator = numbers_to_strings(alloc_mode_val)
                    file_p.write(
                        f"Model_{bench_model_file}_Allocator_{alloc_mode_id}-{allocator}\n"
                    )

                    for ins_idx, ins_val in enumerate(instance):
                        modify_yaml(
                            bench_model_path,
                            framework,
                            ins_val,
                            cores[ins_idx],
                            warmup,
                            iterations,
                            label_file,
                            vocab_file)
                        for ba_s in batch:
                            file_p.write(f"Path {bench_model_path}\n")
                            file_p.write(f"Instance_{ins_val}_Cores_{cores[ins_idx]}_Batch_{ba_s}\n")
                            execute_and_grab(
                                is_performance, bench_model_file, bench_model_path, ba_s, allocator)
                            grab_log(
                                is_performance,
                                bench_model_path,
                                ins_val,
                                cores[ins_idx],
                                file_p)

            file_p.close()
            utils_copy_after_benchmark(bench_model_path)

    except OSError as ex:
        print(ex)

        return 1


def main():
    """parsing user arg."""
    instance_cores = [[4, 7]]
    model_list = [
        "bert_mini_mrpc",
        "distilroberta_base_wnli",
        "distilbert_base_uncased_sst2",
        "roberta_base_mrpc",
        "bert_base_nli_mean_tokens_stsb",
        "bert_base_sparse_mrpc",
        "distilbert_base_uncased_mrpc",
        "bert_mini_sst2",
        "bert_base_mrpc",
        "minilm_l6_h384_uncased_sst2",
        "distilbert_base_uncased_emotion",
        "paraphrase_xlm_r_multilingual_v1_stsb",
        "finbert_financial_phrasebank",
        "bert_large_squad"]

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,)
    parser.add_argument(
        '--batch',
        '-b',
        help="batch size 1,2,3: --batch 1 2 3 ",
        type=int,
        nargs='+',
        dest='batch')
    parser.add_argument(
        '--allocator',
        '-a',
        help="allocator 1,5: --allocator 1 5" +
        "(0:direct 1:cycle,this one is default 2:unified 3:jemalloc+direct 4:jemalloc+cycle " +
        " 5:jemalloc+unified)",
        type=int,
        nargs='+',
        dest='allocator')
    parser.add_argument(
        '--instance_cores',
        '-i',
        help="--instance_cores 4x7 1x28 , it means 4instance 7 cores and 1 instance 28 cores",
        type=str,
        nargs='+',
        dest='i_c')
    parser.add_argument(
        '--model',
        '-m',
        help="--model bert_mini_mrpc,distilbert_base_uncased_sst2,roberta_base_mrpc,"+
        "bert_base_nli_mean_tokens_stsb,bert_base_sparse_mrpc,distilbert_base_uncased_mrpc,"+
        "bert_mini_sst2,bert_base_mrpc,minilm_l6_h384_uncased_sst2,"+
        "distilbert_base_uncased_emotion,paraphrase_xlm_r_multilingual_v1_stsb,"+
        "finbert_financial_phrasebank,bert_large_squad",
        type=str,
        nargs='+',
        dest='model_name')

    parser.add_argument(
        '--warmup',
        '-w',
        help="warmup 10 times: --warmup 10 ",
        type=int,
        dest='warmup')
    parser.add_argument(
        '--iterations',
        '-e',
        help="execute 50 times: --iterations 50 ",
        type=int,
        dest='iterations')
    parser.add_argument(
        '--seq_len',
        '-s',
        help="you can only input one int",
        type=int,
        dest='seq_len')
    parser.add_argument('--int8', type=int, dest='int8')
    parser.add_argument(
        '--is_performance',
        '-p',
        help="1: performance mode, 0: accuracy mode",
        type=int,
        dest='is_performance')
    parser.add_argument(
        '--label_file',
        '-l',
        help="--only bert large need this path",
        type=str,
        dest='label_file')
    parser.add_argument(
        '--vocab_file',
        '-v',
        help="--only bert large need this path",
        type=str,
        dest='vocab_file')
    parser.add_argument(
        '--output_file',
        '-o',
        help="outputfile: --output_file benchmark.txt",
        type=str,
        dest='output_file')

    args = parser.parse_args()
    batch_size = list(args.batch) if args.batch else [16, 32]
    allocator_mode = list(args.allocator) if args.allocator else [1]
    if args.i_c:
        instance_cores = []
        ic_val = []
        for ic_val in args.i_c:
            ic_value = ic_val.split("x")
            tmp_list = [int(ic_value[0]), int(ic_value[1])]
            instance_cores.append(tmp_list)

    if args.model_name:
        model_list = list(args.model_name)
    warmup = args.warmup if args.warmup else 5
    iterations = args.iterations if args.iterations else 10
    is_int8 = args.int8 == 1
    is_performance = args.is_performance != 0
    label_file = args.label_file if args.label_file else ""
    vocab_file = args.vocab_file if args.vocab_file else ""
    output_file = args.output_file if args.output_file else "benchmark.txt"
    sequence_len = args.seq_len if args.seq_len else 0
    test_all(
        is_performance,
        model_list,
        batch_size,
        instance_cores,
        allocator_mode,
        sequence_len,
        warmup,
        iterations,
        is_int8,
        label_file,
        vocab_file,
        output_file)


if __name__ == "__main__":
    main()
