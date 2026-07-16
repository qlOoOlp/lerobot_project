#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""UmiDiffusion config — pose 표현 옵션 + depth 게이트.

═══════════════════════════════════════════════════════════════════════════════
■ 출처 (VENDORED — 손으로 고치기 전에 여기부터 읽을 것)
    lerobot v0.4.4  src/lerobot/policies/diffusion/configuration_diffusion.py
    를 **복사**한 뒤 개명·확장했다. 상속이 아니라 복사다.

    원본과의 차이 (재동기화 시 이것만 다시 얹으면 된다):
      1. @register_subclass("diffusion")        -> "umidiffusion"
      2. class DiffusionConfig(PreTrainedConfig) -> class UmiDiffusionConfig(PreTrainedConfig)
      3. normalization_mapping 기본값: STATE/ACTION MIN_MAX -> IDENTITY  (아래 근거)
      4. 추가 필드: obs_pose_repr / action_pose_repr / use_depth
      5. __post_init__ 끝에 우리 검증 2개 추가
      6. apply_depth_gate() 추가
    재동기화: `diff <(sed -n '17,259p' <lerobot>/policies/diffusion/configuration_diffusion.py) this`

■ ★ 왜 상속(DiffusionConfig)이 아니라 복사인가 — 이게 이 파일의 존재 이유다
    lerobot 의 make_pre_post_processors 는 **isinstance 로 분기**한다:
        factory.py:296   elif isinstance(policy_cfg, DiffusionConfig):
                             make_diffusion_pre_post_processors(...)   <- lerobot 것
        factory.py:394   else:
                             _make_processors_from_policy_config(...)  <- 우리 것 (이름 컨벤션)
    DiffusionConfig 를 상속하면 isinstance 가 True 라 **우리 프로세서 팩토리가 영원히
    안 불린다**. 게다가 조용하다 — lerobot 의 diffusion 프로세서가 대신 일해서 학습이
    "성공"하고, 정책은 anchor-relative 변환 없이 절대 canonical 을 보며 학습한다.
    (실측: 우리 함수 본문이 `...`(None) 인데도 make_pre_post_processors 가 정상 반환했다.)

    lerobot_hong 은 이 벽에 부딪혀 factory.py 에 분기를 끼워넣어 뚫었다. 하지만 lerobot/ 은
    .gitignore 라 그 패치는 버전관리조차 안 되고, 새 머신에서 clone 하면 조용히 재발한다.

    공식 규약(docs/source/bring_your_own_policies.mdx)은 "config 는 PreTrainedConfig 를,
    정책은 PreTrainedPolicy 를 상속하라"고 한다. 공식 예제 lerobot_policy_ditflow 도
    (DiT + flow-matching = diffusion 계열인데도) DiffusionConfig 를 상속하지 않는다.
    즉 기존 정책 config 상속이 규약 이탈이었고, 그 대가가 factory 패치였다.
    상세: refactoring.md 부록 D.7

■ 플러그인 등록 (lerobot 무수정)
    @PreTrainedConfig.register_subclass("umidiffusion") 하나로 두 갈래가 다 열린다:
      정책   : _get_policy_cls_from_policy_name  -> "UmiDiffusionConfig" -"Config" +"Policy"
                                                 -> configuration_ -> modeling_ 치환
      프로세서: _make_processors_from_policy_config -> f"make_{type}_pre_post_processors"
                                                 -> configuration_ -> processor_ 치환
    **단 패키지가 import 되어야 등록이 일어난다** -> __init__.py 에서 export 할 것.
    이 config 가 뿌리다: 클래스명·모듈경로에서 나머지 전부가 문자열 치환으로 유도된다.
═══════════════════════════════════════════════════════════════════════════════
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig

from lerobot_canonical import keys

DEPTH_KEY = keys.DEPTH_KEY


@PreTrainedConfig.register_subclass("umidiffusion")
@dataclass
class UmiDiffusionConfig(PreTrainedConfig):
    """Configuration class for UmiDiffusionPolicy.

    Defaults are configured for training with PushT providing proprioceptive and single camera observations.

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `input_features` and `output_features`.

    Notes on the inputs and outputs:
        - "observation.state" is required as an input key.
        - Either:
            - At least one key starting with "observation.image is required as an input.
              AND/OR
            - The key "observation.environment_state" is required as input.
        - If there are multiple keys beginning with "observation.image" they are treated as multiple camera
          views. Right now we only support all images having the same shape.
        - "action" is required as an output key.

    ■ 우리가 추가한 필드
      obs_pose_repr    : 관측 표현. **"relative" 만** 지원(정책이 관측을 만들지 않으니 backward 가 없음).
      action_pose_repr : 액션 표현. {"relative", "delta"}. 기본 relative.
                         ★ 이 값이 **학습(step)·추론(decode_policy_action) 양쪽**에 흘러야 한다.
                            원본 UMI 는 학습에서 obs_pose_repr 을 잘못 써서 delta 설정 시 조용히 깨짐.
      use_depth        : depth ablation 스위치. metaworld=False(depth 없음), UMI=True.

    ■ ★ normalization_mapping 이 STATE/ACTION = IDENTITY 인 이유 (dev_plan §11)
        원본 DiffusionConfig 는 MIN_MAX 다. 우리가 IDENTITY 로 바꾼 근거:
        dataset stats 는 **canonical(절대)** 기준으로 계산되는데, 런타임 step 이 이를
        **relative** 로 바꾼다 -> canonical stats 로 relative 를 정규화하면 표현 공간이 안 맞는다.
        1차 전략: 정규화하지 않음(relative 값은 이미 0 근처 작은 범위). **이게 유일한 근거.**
        확장: 필요해지면 relative 기준 stats 를 따로 계산.
        ⚠ "metaworld rot6d std=0 나눗셈 회피"는 **근거가 아니다** — 2-0 실측으로 반증됨.
           lerobot 이 `denom = std + eps`(eps=1e-8)로 이미 막는다
           (processor/normalize_processor.py:94, :335). MEAN_STD 여도 NaN 안 남
           (상수 채널은 0/1e-8=0 → 죽은 채로 들어갈 뿐).
      VISUAL 은 default diffusion 전략(MEAN_STD) 유지.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        horizon: Diffusion model action prediction size as detailed in `UmiDiffusionPolicy.select_action`.
        n_action_steps: The number of action steps to run in the environment for one invocation of the policy.
            See `UmiDiffusionPolicy.select_action` for more details.
        input_features: A dictionary defining the PolicyFeature of the input data for the policy. The key represents
            the input data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        output_features: A dictionary defining the PolicyFeature of the output data for the policy. The key represents
            the output data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        normalization_mapping: A dictionary that maps from a str value of FeatureType (e.g., "STATE", "VISUAL") to
            a corresponding NormalizationMode (e.g., NormalizationMode.MIN_MAX)
        vision_backbone: Name of the torchvision resnet backbone to use for encoding images.
        resize_shape: (H, W) shape to resize images to as a preprocessing step for the vision
            backbone. If None, no resizing is done and the original image resolution is used.
        crop_ratio: Ratio in (0, 1] used to derive the crop size from resize_shape
            (crop_h = int(resize_shape[0] * crop_ratio), likewise for width).
            Set to 1.0 to disable cropping. Only takes effect when resize_shape is not None.
        crop_shape: (H, W) shape to crop images to. When resize_shape is set and crop_ratio < 1.0,
            this is computed automatically. Can also be set directly for legacy configs that use
            crop-only (without resize). If None and no derivation applies, no cropping is done.
        crop_is_random: Whether the crop should be random at training time (it's always a center
            crop in eval mode).
        pretrained_backbone_weights: Pretrained weights from torchvision to initialize the backbone.
            `None` means no pretrained weights.
        use_group_norm: Whether to replace batch normalization with group normalization in the backbone.
            The group sizes are set to be about 16 (to be precise, feature_dim // 16).
        spatial_softmax_num_keypoints: Number of keypoints for SpatialSoftmax.
        use_separate_rgb_encoder_per_camera: Whether to use a separate RGB encoder for each camera view.
        down_dims: Feature dimension for each stage of temporal downsampling in the diffusion modeling Unet.
            You may provide a variable number of dimensions, therefore also controlling the degree of
            downsampling.
        kernel_size: The convolutional kernel size of the diffusion modeling Unet.
        n_groups: Number of groups used in the group norm of the Unet's convolutional blocks.
        diffusion_step_embed_dim: The Unet is conditioned on the diffusion timestep via a small non-linear
            network. This is the output dimension of that network, i.e., the embedding dimension.
        use_film_scale_modulation: FiLM (https://huggingface.co/papers/1709.07871) is used for the Unet conditioning.
            Bias modulation is used be default, while this parameter indicates whether to also use scale
            modulation.
        noise_scheduler_type: Name of the noise scheduler to use. Supported options: ["DDPM", "DDIM"].
        num_train_timesteps: Number of diffusion steps for the forward diffusion schedule.
        beta_schedule: Name of the diffusion beta schedule as per DDPMScheduler from Hugging Face diffusers.
        beta_start: Beta value for the first forward-diffusion step.
        beta_end: Beta value for the last forward-diffusion step.
        prediction_type: The type of prediction that the diffusion modeling Unet makes. Choose from "epsilon"
            or "sample". These have equivalent outcomes from a latent variable modeling perspective, but
            "epsilon" has been shown to work better in many deep neural network settings.
        clip_sample: Whether to clip the sample to [-`clip_sample_range`, +`clip_sample_range`] for each
            denoising step at inference time. WARNING: you will need to make sure your action-space is
            normalized to fit within this range.
        clip_sample_range: The magnitude of the clipping range as described above.
        num_inference_steps: Number of reverse diffusion steps to use at inference time (steps are evenly
            spaced). If not provided, this defaults to be the same as `num_train_timesteps`.
        do_mask_loss_for_padding: Whether to mask the loss when there are copy-padded actions. See
            `LeRobotDataset` and `load_previous_and_future_frames` for more information. Note, this defaults
            to False as the original Diffusion Policy implementation does the same.
    """

    # ── 우리 필드 (원본에 없음) ─────────────────────────────────────────────
    obs_pose_repr: str = "relative"
    action_pose_repr: str = "relative"
    use_depth: bool = True

    # Inputs / output structure.
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    # ★ 원본과 다름: STATE/ACTION 이 MIN_MAX -> IDENTITY (클래스 docstring 근거 참고)
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    # The original implementation doesn't sample frames for the last 7 steps,
    # which avoids excessive padding and leads to improved training results.
    drop_n_last_frames: int = 7  # horizon - n_action_steps - n_obs_steps + 1

    # Architecture / modeling.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    resize_shape: tuple[int, int] | None = None
    crop_ratio: float = 1.0
    crop_shape: tuple[int, int] | None = None
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False
    # Unet.
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True
    # Noise scheduler.
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # Inference
    num_inference_steps: int | None = None

    # Optimization
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"

    # Loss computation
    do_mask_loss_for_padding: bool = False

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )

        supported_prediction_types = ["epsilon", "sample"]
        if self.prediction_type not in supported_prediction_types:
            raise ValueError(
                f"`prediction_type` must be one of {supported_prediction_types}. Got {self.prediction_type}."
            )
        supported_noise_schedulers = ["DDPM", "DDIM"]
        if self.noise_scheduler_type not in supported_noise_schedulers:
            raise ValueError(
                f"`noise_scheduler_type` must be one of {supported_noise_schedulers}. "
                f"Got {self.noise_scheduler_type}."
            )

        if self.resize_shape is not None and (
            len(self.resize_shape) != 2 or any(d <= 0 for d in self.resize_shape)
        ):
            raise ValueError(f"`resize_shape` must be a pair of positive integers. Got {self.resize_shape}.")
        if not (0 < self.crop_ratio <= 1.0):
            raise ValueError(f"`crop_ratio` must be in (0, 1]. Got {self.crop_ratio}.")

        if self.resize_shape is not None:
            if self.crop_ratio < 1.0:
                self.crop_shape = (
                    int(self.resize_shape[0] * self.crop_ratio),
                    int(self.resize_shape[1] * self.crop_ratio),
                )
            else:
                # Explicitly disable cropping for resize+ratio path when crop_ratio == 1.0.
                self.crop_shape = None
        if self.crop_shape is not None and (self.crop_shape[0] <= 0 or self.crop_shape[1] <= 0):
            raise ValueError(f"`crop_shape` must have positive dimensions. Got {self.crop_shape}.")

        # Check that the horizon size and U-Net downsampling is compatible.
        # U-Net downsamples by 2 with each stage.
        downsampling_factor = 2 ** len(self.down_dims)
        if self.horizon % downsampling_factor != 0:
            raise ValueError(
                "The horizon should be an integer multiple of the downsampling factor (which is determined "
                f"by `len(down_dims)`). Got {self.horizon=} and {self.down_dims=}"
            )

        # ── 우리 검증 (원본에 없음) ─────────────────────────────────────────
        if self.obs_pose_repr != "relative":
            raise ValueError(
                f'`obs_pose_repr` must be "relative" (the only supported value). Got {self.obs_pose_repr}.'
            )
        if self.action_pose_repr not in {"relative", "delta"}:
            raise ValueError(
                f'`action_pose_repr` must be one of {{"relative", "delta"}}. Got {self.action_pose_repr}.'
            )

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

        if self.resize_shape is None and self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` should fit within the image shapes. Got {self.crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for `{key}`."
                    )

        # Check that all input images have the same shape.
        if len(self.image_features) > 0:
            first_image_key, first_image_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_image_ft.shape:
                    raise ValueError(
                        f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                    )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def apply_depth_gate(self) -> None:
        """use_depth=False 면 input_features 에서 depth 키를 제거한다.

        ■ 이게 lerobot 패치 60줄을 대체한다
          lerobot_hong 은 datasets/factory.py(+15) + policies/factory.py(+45) 를 패치해
          depth 를 걸러냈다. 그 로직을 **config 로 옮겨** 무수정을 달성한다.

        ■ 원리
          depth 를 input_features 에서 빼면 UmiDiffusionPolicy 가 **depth 인코더를 안 만들고**
          배치의 depth 를 무시한다 -> 별도 필터 불필요.

        ■ hook 위치가 중요
          make_policy 는 input_features 를 채운 뒤(factory.py:517) validate_features 를
          **안 부르고** 바로 정책을 생성한다 => 필터는 **UmiDiffusionPolicy.__init__ 의 super() 직전**이
          정답(이때 input_features 는 세팅됐고 모델은 아직 안 만들어짐).

        ■ 유의
          - **idempotent** 해야 한다(정책/프로세서 어느 쪽이 먼저 불러도 동일). 이미 없으면 no-op.
          - input_features 가 비어있을 수 있으니 방어.
          - 체크포인트 config 에 필터된 input_features 가 저장되므로 로드 시에도 일관.
          - ⚠ depth 게이트는 **두 겹**: 이 config 게이트(모델이 인코더를 안 만들게) +
            DropObservationKeysProcessorStep(관측 dict 에서 실제 제거). 둘 다 필요.
        """
        ...  # 구현 ②
