# -*- coding: utf-8 -*-
"""
自主科研 Agent: Literature -> Diagnosis -> Design -> Experiment -> Memory -> Decision.
"""

import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Callable


@dataclass
class ExperimentResult:
    """单次实验结果。"""

    config: dict
    metric: float = 0.0
    metric_name: str = 'val_accuracy'
    train_time: float = 0.0
    status: str = 'success'
    error: str = ''
    timestamp: str = ''
    round_num: int = 0


@dataclass
class Diagnosis:
    """诊断结果。"""

    observations: list[str]
    hypotheses: list[str]
    trend: str


class ResearchAgent:
    """自主科研 Agent。

    闭环迭代: Literature -> Diagnosis -> Design -> Experiment -> Memory -> Decision

    决策: CONTINUE / PIVOT / STOP
    """

    def __init__(
        self,
        task_type: str,
        run_fn: Callable,
        llm_client=None,
        memory_path: str = 'output/research_memory.json',
    ):
        self.task_type = task_type
        self.run_fn = run_fn
        self.llm_client = llm_client
        self.memory_path = memory_path
        self.memory = []
        self.hypotheses = []
        self.budget_remaining = 7200
        self.round_num = 0
        self.max_rounds = 10

    def run(self, initial_config):
        self._load_memory()
        print(chr(10) + "=" * 60)
        print('第1轮: 冷启动')
        print("=" * 60)
        self._literature_phase(initial_config)
        self.round_num += 1
        result = self._experiment_phase(initial_config)
        self._memory_phase(result)
        self._diagnosis_phase()
        for _ in range(1, self.max_rounds):
            if self.budget_remaining < 60:
                print(chr(10) + '预算耗尽, 停止')
                break
            print(chr(10) + "=" * 60)
            print(f"第{self.round_num + 1}轮: 迭代优化")
            self._diagnosis_phase()
            config = self._design_phase()
            self.round_num += 1
            result = self._experiment_phase(config)
            self._memory_phase(result)
            decision = self._decision_phase()
            if decision == "STOP":
                break
        return self._summarize()

    def _literature_phase(self, config):
        print('[Literature] 分析数据和代码...')
        print(f"  任务类型: {self.task_type}")
        print(f"  初始配置: {config}")
        if self.llm_client and self.llm_client.available:
            prompt = f'分析以下实验配置，提出初始假设: 任务={self.task_type} 配置={json.dumps(config)}'
            try:
                resp = self.llm_client.chat('你是一个机器学习研究专家。', prompt)
                print(f'  LLM 建议: {resp[:100]}...')
            except Exception as e:
                print(f'  LLM 调用失败: {e}')

    def _diagnosis_phase(self):
        print('[Diagnosis] 分析历史记录...')
        if not self.memory:
            return
        recent = self.memory[-5:] if len(self.memory) >= 5 else self.memory
        metrics = [r.metric for r in recent]
        if len(metrics) >= 2:
            if metrics[-1] > metrics[0]:
                self.hypotheses = ['继续优化', '尝试更大容量', '调整学习率']
            elif metrics[-1] < metrics[0] - 0.01:
                self.hypotheses = ['回退最佳配置', '减少复杂度', '增加正则化']
            else:
                self.hypotheses = ['尝试新架构', '特征工程', '调整搜索空间']
        print(f"  假设: {self.hypotheses}")

    def _design_phase(self):
        print('[Design] 生成新配置...')
        if not self.memory:
            return self._default_config()
        if self.llm_client and self.llm_client.available:
            try:
                config = self._llm_design_phase()
                if config:
                    return config
            except Exception as e:
                print(f'  LLM 设计失败: {e}')
        best = max(self.memory, key=lambda r: r.metric)
        config = self._perturb_config(best.config)
        return config

    def _llm_design_phase(self):
        from llm.prompts import SYSTEM_PROMPT, HPO_PROMPT
        from llm.parser import parse_structured
        from llm.schemas import HPOSuggestion
        best = max(self.memory, key=lambda r: r.metric)
        best_config = best.config
        best_metric = best.metric
        hist_lines = []
        for r in self.memory:
            m = f"{r.metric_name}={r.metric:.4f}"
            c = json.dumps(r.config, ensure_ascii=False)
            hist_lines.append(f"  Round {r.round_num}: {m} | status={r.status} | config={c}")
        history_str = chr(10).join(hist_lines) if hist_lines else '无'
        if self.task_type == "classification":
            search_space = {"n_estimators": [300,400,500,600,800], "max_depth": [5,6,7,8,10], "learning_rate": [0.01,0.02,0.03,0.05], "subsample": [0.6,0.7,0.8,0.9,1.0], "colsample_bytree": [0.6,0.7,0.8,0.9,1.0], "reg_alpha": [0.0,0.01,0.1,0.5,1.0], "reg_lambda": [0.0,0.01,0.1,0.5,1.0], "min_child_samples": [5,10,20,30,50], "hidden_dim": [128,192,256,384], "gcnii_layers": [8,12,16,24,32], "dropout": [0.3,0.4,0.5,0.6], "lr": [0.005,0.008,0.01,0.02], "weight_decay": [1e-4,5e-4,1e-3]}
        else:
            search_space = {"embedding_dim": [16,32,64,128,256], "lr": [1e-5,5e-5,1e-4,5e-4,1e-3]}
        d = '; '.join(self.hypotheses) if self.hypotheses else '无'
        up = HPO_PROMPT.format(task_type=self.task_type, history=history_str, best_config=json.dumps(best_config, indent=2, ensure_ascii=False), best_metric=f'{best.metric_name}={best_metric:.4f}', search_space=json.dumps(search_space, indent=2, ensure_ascii=False), diagnosis=d)
        resp = self.llm_client.chat(SYSTEM_PROMPT, up)
        sug = parse_structured(resp, HPOSuggestion)
        if not sug:
            print('  LLM 响应解析失败')
            return None
        print(f"  LLM 推理: {sug.reasoning[:80]}...")
        print(f"  LLM 修改: {sug.config_updates}")
        print(f"  LLM 置信度: {sug.confidence:.2f}")
        new_cfg = best_config.copy()
        for k, v in sug.config_updates.items():
            new_cfg[k] = v
        return new_cfg

    def _experiment_phase(self, config):
        print('[Experiment] 训练 ' + str(config.get('model_type', 'unknown')) + '...')
        start_time = time.time()
        try:
            result = self.run_fn(config)
            result.train_time = time.time() - start_time
            result.timestamp = datetime.now().isoformat()
            result.round_num = self.round_num
            self.budget_remaining -= result.train_time
            return result
        except Exception as e:
            return ExperimentResult(
                config=config,
                metric=0.0,
                metric_name='error',
                train_time=time.time() - start_time,
                status='failed',
                error=str(e),
                timestamp=datetime.now().isoformat(),
                round_num=self.round_num,
            )

    def _memory_phase(self, result):
        self.memory.append(result)
        self._save_memory()
        if result.status == 'success':
            print(f"  结果: {result.metric_name}={result.metric:.4f}")
        else:
            print(f"  失败: {result.error[:50]}")

    def _decision_phase(self):
        print('[Decision] 决策中...')
        if self.budget_remaining < 60:
            print('  -> STOP: 预算耗尽')
            return "STOP"
        if self.round_num >= self.max_rounds - 1:
            print("  -> STOP: 达到最大轮次")
            return "STOP"
        print('  -> CONTINUE')
        return "CONTINUE"

    def _default_config(self):
        if self.task_type == "classification":
            return {
                "model_type": "Ensemble",
                "n_estimators": 500,
                "max_depth": 7,
                "learning_rate": 0.03,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "min_child_samples": 20,
                "hidden_dim": 256,
                "gcnii_layers": 16,
                "dropout": 0.5,
                "lr": 0.01,
                "weight_decay": 5e-4,
            }
        else:
            return {
                "model_type": "Popularity",
            }

    def _perturb_config(self, config):
        import random
        cfg = config.copy()
        if self.task_type == "classification":
            cfg["model_type"] = "Ensemble"
            params = [
                "n_estimators", "max_depth", "learning_rate",
                "subsample", "colsample_bytree", "reg_alpha",
                "reg_lambda", "min_child_samples", "hidden_dim",
                "gcnii_layers", "dropout", "lr", "weight_decay",
            ]
            k = random.randint(2, 4)
            chosen = random.sample(params, k)
            for param in chosen:
                if param == "n_estimators":
                    cfg["n_estimators"] = random.choice([300, 400, 500, 600, 800])
                elif param == "max_depth":
                    cfg["max_depth"] = random.choice([5, 6, 7, 8, 10])
                elif param == "learning_rate":
                    cfg["learning_rate"] = random.choice([0.01, 0.02, 0.03, 0.05])
                elif param == "subsample":
                    cfg["subsample"] = random.choice([0.6, 0.7, 0.8, 0.9, 1.0])
                elif param == "colsample_bytree":
                    cfg["colsample_bytree"] = random.choice([0.6, 0.7, 0.8, 0.9, 1.0])
                elif param == "reg_alpha":
                    cfg["reg_alpha"] = random.choice([0.0, 0.01, 0.1, 0.5, 1.0])
                elif param == "reg_lambda":
                    cfg["reg_lambda"] = random.choice([0.0, 0.01, 0.1, 0.5, 1.0])
                elif param == "min_child_samples":
                    cfg["min_child_samples"] = random.choice([5, 10, 20, 30, 50])
                elif param == "hidden_dim":
                    cfg["hidden_dim"] = random.choice([128, 192, 256, 384])
                elif param == "gcnii_layers":
                    cfg["gcnii_layers"] = random.choice([8, 12, 16, 24, 32])
                elif param == "dropout":
                    cfg["dropout"] = random.choice([0.3, 0.4, 0.5, 0.6])
                elif param == "lr":
                    cfg["lr"] = random.choice([0.005, 0.008, 0.01, 0.02])
                elif param == "weight_decay":
                    cfg["weight_decay"] = random.choice([1e-4, 5e-4, 1e-3, 5e-3])
        else:
            if "embedding_dim" in cfg:
                cfg["embedding_dim"] = random.choice([32, 64, 128])
            if "lr" in cfg:
                cfg["lr"] = random.choice([1e-4, 5e-4, 1e-3])
        return cfg

    def _save_memory(self):
        data = {
            "task_type": self.task_type,
            "round_num": self.round_num,
            "budget_remaining": self.budget_remaining,
            "experiments": [asdict(r) for r in self.memory],
        }
        os.makedirs(os.path.dirname(self.memory_path) if os.path.dirname(self.memory_path) else '.', exist_ok=True)
        f = open(self.memory_path, "w", encoding="utf-8")
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.close()

    def _load_memory(self):
        if os.path.exists(self.memory_path):
            try:
                f = open(self.memory_path, "r", encoding="utf-8")
                data = json.load(f)
                f.close()
                self.memory = [ExperimentResult(**r) for r in data.get("experiments", [])]
                self.round_num = data.get("round_num", 0)
                self.budget_remaining = data.get("budget_remaining", 7200)
                print(f"  加载记忆: {len(self.memory)} 轮历史")
            except Exception:
                pass

    def _summarize(self):
        if not self.memory:
            return {"best_metric": 0.0, "num_rounds": 0}
        successful = [r for r in self.memory if r.status == "success"]
        if not successful:
            return {"best_metric": 0.0, "num_rounds": len(self.memory)}
        best = max(successful, key=lambda r: r.metric)
        return {
            "best_metric": best.metric,
            "best_config": best.config,
            "num_rounds": len(self.memory),
            "best_round": best.round_num,
        }
