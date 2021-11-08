import torch
import math
import time
from transformers.debug_utils import DebugOption
from transformers.trainer_utils import speed_metrics
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
from torch.utils import mkldnn as mkldnn_utils
import sys
if "--use_ipex" in sys.argv and sys.argv[sys.argv.index("--use_ipex") + 1] == "yes":
    import intel_extension_for_pytorch as ipex
if "--jit_mode" in sys.argv and sys.argv[sys.argv.index("--jit_mode") + 1] == "yes":
    if "--use_ipex" in sys.argv and sys.argv[sys.argv.index("--use_ipex") + 1] == "yes":
        #ipex.enable_onednn_fusion(False)
        pass

from transformers import Trainer, is_torch_tpu_available
from transformers.trainer_utils import PredictionOutput

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
from tqdm import tqdm, trange

class TextClassificationTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def evaluate(
        self,
        use_ipex=None,
        jit_mode=None,
        bf16: bool = False,
        back_arch=None,
        instance_id=-1, 
        num_instances=-1, 
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        self._memory_tracker.start()

        if instance_id >= 0 and num_instances >=0:
            sub_eval_dataset = get_dataset_subset(eval_dataset, num_instances, instance_id)
            eval_dataloader = self.get_eval_dataloader(sub_eval_dataset)
        else:
            eval_dataloader = self.get_eval_dataloader(eval_dataset)


        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        if jit_mode:
            dummy_tensor_labels = torch.ones((eval_dataloader.batch_size, 1), dtype=torch.long)
            dummy_tensor_attention_mask = torch.ones((eval_dataloader.batch_size, 128), dtype=torch.long)
            dummy_tensor_postion_id = torch.ones((12), dtype=torch.long)
            dummy_tensor_input_ids = torch.ones((eval_dataloader.batch_size, 128), dtype=torch.long)
            dummy_tensor_token_type_ids = torch.ones((eval_dataloader.batch_size, 128), dtype=torch.long)
            jit_inputs_bert = (dummy_tensor_labels, dummy_tensor_attention_mask, dummy_tensor_input_ids, dummy_tensor_token_type_ids)
            jit_inputs_distilbert = (dummy_tensor_labels, dummy_tensor_attention_mask, dummy_tensor_input_ids)
            jit_inputs_roberta = (dummy_tensor_labels, dummy_tensor_attention_mask, dummy_tensor_input_ids)

            if back_arch == "BERT":
                jit_inputs = jit_inputs_bert
            elif back_arch == "RoBERTa":
                jit_inputs = jit_inputs_roberta
            elif back_arch == "DistilBERT":
                jit_inputs = jit_inputs_distilbert
            else:
                jit_inputs = None
            if use_ipex:
                if bf16:
                    print('testing jit ipex bf16')
                    self.model = ipex.optimize(self.model, dtype=torch.bfloat16, level="O1")
                    with torch.cpu.amp.autocast():
                        self.model = torch.jit.trace(self.model, jit_inputs, strict=False )
                        # self.model = torch.jit.script(self.model)
                    self.model = torch.jit.freeze(self.model)
                else: 
                    print('testing jit ipex fp32')
                    self.model = ipex.optimize(self.model, dtype=torch.float32, level="O1")
                    self.model = torch.jit.trace(self.model, jit_inputs, strict=False)
                    self.model = torch.jit.freeze(self.model)
            else:
                if bf16:
                    print("testing jit stock pt bf16")
                    with torch.cpu.amp.autocast():
                        self.model = torch.jit.trace(self.model, jit_inputs, strict=False)
                    self.model = torch.jit.freeze(self.model)
                else:
                    print("testing jit stock pt fp32")
                    self.model = torch.jit.trace(self.model, jit_inputs, strict=False)
                    self.model = torch.jit.freeze(self.model)
        else:
            if use_ipex:
                if bf16:
                    print("testing imperative ipex bf16")
                    self.model = ipex.optimize(self.model, dtype=torch.bfloat16, level="O1")
                else: 
                    print("testing imperative ipex fp32")
                    self.model = ipex.optimize(self.model, dtype=torch.float32, level="O1")
            else:
                if bf16:
                    print("testing imperative stock pt bf16")
                    self.model = mkldnn_utils.to_mkldnn(self.model.to(memory_format=torch.channels_last), dtype=torch.bfloat16) 
                else:
                    print("testing imperative stock pt fp32")
                    # self.model = mkldnn_utils.to_mkldnn(self.model.to(memory_format=torch.channels_last))
                    pass

        if bf16:
            with torch.cpu.amp.autocast():
                for _,batch in enumerate(eval_dataloader):
                    for _,label in enumerate(batch):
                        if batch[label].dim() >=4:
                            batch[label]=batch[label].to(memory_format=torch.channels_last)
                try:
                    print("tanglei before evaluation")
                    output = eval_loop(
                        eval_dataloader,
                        description="Evaluation",
                        # No point gathering the predictions if there are no metrics, otherwise we defer to
                        # self.args.prediction_loss_only
                        prediction_loss_only=True if self.compute_metrics is None else None,
                        ignore_keys=ignore_keys,
			metric_key_prefix=metric_key_prefix,
                    )
                finally:
                    self.compute_metrics = self.compute_metrics
        else:
            for _,batch in enumerate(eval_dataloader):
                for _,label in enumerate(batch):
                    if batch[label].dim() >= 4:
                        batch[label] = batch[label].to(memory_format=torch.channels_last)
            try:
                print("tanglei before evaluation")
                output = eval_loop(
                    eval_dataloader,
                    description="Evaluation",
                    # No point gathering the predictions if there are no metrics, otherwise we defer to
                    # self.args.prediction_loss_only
                    prediction_loss_only=True if self.compute_metrics is None else None,
                    ignore_keys=ignore_keys
                )
            finally:
                self.compute_metrics = self.compute_metrics

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics
