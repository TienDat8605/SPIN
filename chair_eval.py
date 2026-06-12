import argparse
import json
from tqdm import tqdm

import torch

from attentionPAI import llama_modify
from attentionSPIN import llama_modify_spin, llama_modify_cb

from constants import INSTRUCTION_TEMPLATE, SYSTEM_MESSAGE
from eval_data_loader import COCODataSet, POPEChatCalibDataset
from llava.utils import disable_torch_init
from model_loader import ModelLoader

from CFG_damro import DamroCFGLogits
from utils import setup_seeds, add_diffusion_noise

from calibrate_spectral import calibrate_spectral
from spectral_config import SpectralConfig

from transformers.generation.logits_process import LogitsProcessorList

parser = argparse.ArgumentParser(description="CHAIR evaluation on LVLMs.")
parser.add_argument("--model", type=str, help="model: llava-1.5, minigpt4, shikra")
parser.add_argument(
    "--data-path",
    type=str,
    default="/path/to/coco/val2014/",
    help="data path",
)

parser.add_argument("--llava-size", type=str, default='7b', help="7b or 13b")

parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--beam", type=int, default=1)
parser.add_argument("--sample", action="store_true")
parser.add_argument("--max-tokens", type=int, default=512)

parser.add_argument("--start-layer", type=int, default=0)
parser.add_argument("--end-layer", type=int, default=32)

# -------------------- PAI -------------------
parser.add_argument("--use-pai", action="store_true", help='Use PAI or not')
parser.add_argument("--use-attn", action="store_true")
parser.add_argument("--alpha", type=float, default=0.2)
parser.add_argument("--use-cfg", action="store_true")
parser.add_argument("--gamma", type=float, default=2)
# --------------------------------------------

# ------------------- SPIN -------------------
parser.add_argument("--use-spin", action="store_true", help='Use SPIN or not')
parser.add_argument("--routed-heads", type=float, default=0.95, 
                    help='Fraction of heads that is activated for SPIN')
parser.add_argument("--small-num-mask", type=float, default=None,
                    help="The scaling factor for SPIN")
parser.add_argument("--repetition-penalty", type=float, default=1.0,
                    help="Set to 1.1 when using Minigpt4")
# --------------------------------------------

# --------------- CB-Spectral -----------------
parser.add_argument("--use-cb", action="store_true", help="Use CB-Spectral head modulation (overrides --use-spin when set)")
parser.add_argument("--spectral-mode", type=str, default="fft", choices=["fft", "power", "block", "none"],
                    help="Spectral feature extraction mode. 'none' = legacy SPIN pruning.")
parser.add_argument("--suppression-coeff", type=float, default=0.3)
parser.add_argument("--reinforcement-coeff", type=float, default=0.15)
parser.add_argument("--temperature", type=float, default=0.1)
parser.add_argument("--n-calib-examples", type=int, default=64)
parser.add_argument("--calib-pope-type", type=str, default="random", choices=["random", "popular", "adversarial"])
# --------------------------------------------

# ---------------- DAMRO ----------------
parser.add_argument("--use-damro", action="store_true", help='Use DAMRO or not')
parser.add_argument("--alpha-damro", type=float, default=0.2,
                    help='alpha for DAMRO logits refinement')
parser.add_argument("--outlier-topk", type=int, default=10,
                    help='Number of outliers to be removed')
# --------------------------------------

# ------------------- OPERA -------------------
parser.add_argument("--use-opera", action="store_true", help='Use OPERA or not')
parser.add_argument("--scale-factor", type=float, default=50.0)
parser.add_argument("--threshold", type=int, default=15)
parser.add_argument("--num-candidates", type=int, default=5)
parser.add_argument("--penalty-weights", type=float, default=1.0)
# ---------------------------------------------

# ------------------- VCD -------------------
parser.add_argument("--use-cd", action='store_true', help='Use VCD or not')
parser.add_argument("--noise-step", type=int, default=999)
parser.add_argument("--cd-alpha", type=float, default=1.0)
parser.add_argument("--cd-beta", type=float, default=0.1)
# -------------------------------------------

parser.add_argument("--output-path", type=str, default="", help="saving CHAIR results.")
args = parser.parse_known_args()[0]

assert args.model in ['llava-1.5', 'minigpt4', 'shikra'], (
    f'We support llava-1.5, minigpt4, and shikra. But got {args.model}. '
    'If you want to use Qwen-VL-Chat, please go to chair_eval_qwen.py'
)
if not args.output_path:
    print('You are running the script without saving the outputs.')
if args.model == 'minigpt4' and args.use_cd:
    raise ValueError('VCD is not implemented for Q-former based model like Minigpt4.')
if args.model != 'llava-1.5' and args.use_damro:
    raise ValueError('DAMRO only implemented for llava.')

setup_seeds()
disable_torch_init()

model_loader = ModelLoader(args.model, args.llava_size)

if args.use_cb and not model_loader.is_cb_spectral_compatible():
    model_loader.warn_if_cb_incompatible()
    args.use_cb = False
    args.spectral_mode = "none"

cb_thresholds = None
if args.use_cb and args.spectral_mode != "none":
    spectral_cfg = SpectralConfig(
        spectral_mode=args.spectral_mode,
        suppression_coeff=args.suppression_coeff,
        reinforcement_coeff=args.reinforcement_coeff,
        temperature=args.temperature,
        n_calib_examples=args.n_calib_examples,
    )
    calib_dataset = POPEChatCalibDataset(
        pope_path=f"./pope_coco/chat/coco_pope_chat_{args.calib_pope_type}.json",
        data_path=args.data_path,
        trans=model_loader.image_processor,
        max_examples=args.n_calib_examples,
    )
    questions_dummy = ["Please describe the image in detail."]
    image_dummy = next(iter(torch.utils.data.DataLoader(calib_dataset, batch_size=1)))[
        "image"
    ]
    _, img_s, img_e, _ = model_loader.prepare_inputs_for_model(template, questions_dummy, image_dummy)
    cb_thresholds = calibrate_spectral(
        model=model_loader.llm_model,
        dataset=calib_dataset,
        template=template,
        prepare_inputs_fn=model_loader.prepare_inputs_for_model,
        spectral_cfg=spectral_cfg,
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        img_start_idx=img_s,
        img_end_idx=img_e,
        batch_size=args.batch_size,
    )
    print(f"[chair_eval] Calibrated per-layer thresholds: {cb_thresholds}")

coco_dataset = COCODataSet(data_path=args.data_path, trans=model_loader.image_processor)
coco_loader = torch.utils.data.DataLoader(
    coco_dataset, batch_size=args.batch_size, shuffle=False, num_workers=32
)
    
template = INSTRUCTION_TEMPLATE[args.model]
if args.model == "llava-1.5" or args.model == "shikra":
    template = SYSTEM_MESSAGE + template

for batch_id, data in tqdm(enumerate(coco_loader), total=len(coco_loader)):
    if batch_id == 500:
        break

    img_id = data["img_id"]
    image = data["image"]

    batch_size = img_id.shape[0]
    query = ["Please help me describe the image in detail."] * batch_size
    questions, kwargs = model_loader.prepare_inputs_for_model(template, query, image)

    # ------------------- VCD -------------------
    if args.use_cd:
        image_tensor_cd = add_diffusion_noise(kwargs["images"], args.noise_step)
        kwargs['images_cd'] = image_tensor_cd
        kwargs['cd_alpha'] = args.cd_alpha
        kwargs['cd_beta'] = args.cd_beta
    else:
        image_tensor_cd = None
    # -------------------------------------------

    # ------------------- OPERA -----------------
    if args.use_opera:
        key_position = {
            "image_start": model_loader.img_start_idx, 
            "image_end": model_loader.img_end_idx - 1,
            "response_start": model_loader.response_start_idx,
        }
        kwargs['opera_decoding'] = True
        kwargs['key_position'] = key_position
        kwargs['scale_factor'] = args.scale_factor
        kwargs['threshold'] = args.threshold
        kwargs['num_attn_candidates'] = args.num_candidates
        kwargs['penalty_weights'] = args.penalty_weights
        output_attentions = True  # Optional. But required for OPERA
    else:
        kwargs['opera_decoding'] = False
        kwargs['key_position'] = None
        output_attentions = False  # You can set it to True if you want
    # -------------------------------------------

    if args.use_pai:
        llama_modify(
            model_loader.llm_model,
            args.start_layer,
            args.end_layer,
            args.use_attn,
            args.alpha,
            args.use_cfg,
            model_loader.img_start_idx,
            model_loader.img_end_idx,
        )

        logits_processor = (
            model_loader.init_cfg_processor(questions, args.gamma, args.beam, args.start_layer, args.end_layer)
            if args.use_cfg
            else None
        )

        if logits_processor is not None:
            kwargs["logits_processor"] = LogitsProcessorList([logits_processor])

    elif args.use_spin:
        llama_modify_spin(
            model_loader.llm_model,
            args.start_layer,
            args.end_layer,
            model_loader.img_start_idx,
            model_loader.img_end_idx,
            routed_head=args.routed_heads,
            use_spin_img=True,
            small_num_mask=args.small_num_mask,
        )

    elif args.use_cb:
        spectral_cfg = SpectralConfig(
            spectral_mode=args.spectral_mode,
            suppression_coeff=args.suppression_coeff,
            reinforcement_coeff=args.reinforcement_coeff,
            temperature=args.temperature,
            n_calib_examples=args.n_calib_examples,
        )
        n_gated = args.end_layer - args.start_layer
        if cb_thresholds is not None:
            tau_weak = torch.tensor(
                [cb_thresholds[l][0] for l in range(n_gated)],
                dtype=torch.float32,
            )
            tau_strong = torch.tensor(
                [cb_thresholds[l][1] for l in range(n_gated)],
                dtype=torch.float32,
            )
        else:
            tau_weak = None
            tau_strong = None
        llama_modify_cb(
            model_loader.llm_model,
            args.start_layer,
            args.end_layer,
            model_loader.img_start_idx,
            model_loader.img_end_idx,
            spectral_cfg=spectral_cfg,
            tau_weak=tau_weak,
            tau_strong=tau_strong,
            capture=False,
        )
    
    elif args.use_damro:
        logits_processor = DamroCFGLogits(args.alpha_damro, model_loader.llm_model, images=kwargs["images"])
        kwargs["logits_processor"] = LogitsProcessorList([logits_processor])

    with torch.inference_mode():
        model_loader.llm_model.generation_config.repetition_penalty = args.repetition_penalty
        model_loader.llm_model.generation_config.temperature = 1.0  # originally 1.0
        model_loader.llm_model.generation_config.top_p = 1.0  # originally 1.0
        model_loader.llm_model.generation_config.top_k = None  # originally 50

        outputs = model_loader.llm_model.generate(
            do_sample=args.sample,
            max_new_tokens=args.max_tokens,
            use_cache=True,
            num_beams=args.beam,
            output_attentions=output_attentions,
            output_hidden_states=False,
            return_dict=True,
            **kwargs,
        )

    output_text = model_loader.decode(outputs)

    if args.output_path:
        for i in range(len(output_text)):
            with open(args.output_path, "a", encoding='utf-8') as f:
                json.dump({"image_id": int(img_id[i]), "caption": output_text[i]}, f)
                f.write("\n")
