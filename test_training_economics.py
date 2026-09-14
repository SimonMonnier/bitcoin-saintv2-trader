import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
import tempfile
import training as t
import saint_core as core
import numpy as np
import pandas as pd
from datetime import datetime
import importlib


def environment(**kwargs):
    cfg=t.PPOConfig(episode_length=100, use_vol_curriculum=False,
        spread_bps=0., spread_wide_prob=0., entry_slippage_bps=0.,
        slippage_bps=0., tick_noise_bps=0., fee_rate=0., use_be_trail=False)
    for k,v in kwargs.items(): setattr(cfg,k,v)
    n=400
    data=SimpleNamespace(features=np.zeros((n,t.OBS_N_FEATURES-4),np.float32),
        close=np.full(n,10000.), open=np.full(n,10000.), high=np.full(n,10000.),
        low=np.full(n,10000.), atr14=np.full(n,100.), length=n)
    env=t.BTCTradingEnvDiscrete(data,cfg)
    env.idx=env.start_idx=60;env.end_idx=160
    env._sample_trade_spread_bps=lambda: cfg.spread_bps
    return env


class EconomicsTests(unittest.TestCase):
    def test_backtest_entries_share_bid_ask_convention(self):
        for name in ('backtest_saintv2_no_be_trail', 'backtest_saintv2_stress_test', 'backtest_saintv2_multi_agent'):
            with self.subTest(module=name):
                module=importlib.import_module(name)
                cfg=module.LiveConfig();cfg.spread_bps=.001
                stress=module.StressConfig();stress.enable=False
                buy,spread=module.compute_execution_price(1,10000,datetime(2026,1,1,12),cfg,stress,True)
                sell=module.compute_execution_price(-1,10000,datetime(2026,1,1,12),cfg,stress)
                self.assertAlmostEqual(buy,10010)
                self.assertEqual(sell,10000)
                self.assertEqual(spread,.001)

    def test_long_no_early_stop_and_no_double_spread(self):
        e=environment(spread_bps=10.)
        e.step(0)
        self.assertAlmostEqual(e.entry_price,10010.)
        stop=e.sl_price;size=e.current_size
        e.data.low[e.idx]=stop+5
        e.step(2)
        self.assertEqual(len(e.trades_pnl),0)
        e.data.low[e.idx]=stop-1
        e.step(2)
        self.assertAlmostEqual(e.trades_meta[-1]['exit_price'],stop)
        self.assertAlmostEqual(e.trades_pnl[-1],(stop-10010)*size)

    def test_short_stop_uses_ask(self):
        e=environment(spread_bps=10.)
        e.step(1)
        self.assertEqual(e.entry_price,10000.)
        stop=e.sl_price
        e.data.high[e.idx]=stop/1.001-1
        e.step(2)
        self.assertEqual(len(e.trades_pnl),0)
        e.data.high[e.idx]=stop/1.001+1
        e.step(2)
        self.assertAlmostEqual(e.trades_meta[-1]['exit_price'],stop)

    def test_constant_price_noise_does_not_create_loss(self):
        e=environment(tick_noise_bps=1.2)
        with patch.object(t.np.random,'uniform',side_effect=lambda a,b:(a+b)/2):
            rewards=[e.step(0 if k==0 else 2)[1] for k in range(60)]
        np.testing.assert_allclose(rewards,0,atol=1e-12)

    def test_terminal_liquidation_both_sides_counts_cost_once(self):
        for action in (0,1):
            with self.subTest(action=action):
                e=environment(spread_bps=10.,fee_rate=.0004)
                first=e.step(action)[1];size=e.current_size
                e.end_idx=e.idx+1
                _,reward,done,truncated,info=e.step(2)
                self.assertTrue(done);self.assertFalse(truncated)
                self.assertEqual(info['position'],0)
                self.assertEqual(len(e.trades_pnl),1)
                quote=10000 if action==0 else 10010
                expected=(-10-.0004*quote)*size
                self.assertAlmostEqual(e.capital-1000,expected)
                self.assertAlmostEqual(sum(e.trades_pnl),expected)
                self.assertLess(first,0)
                self.assertLess(reward,0)
                self.assertEqual(e.trades_meta[0]['terminal_reason'],'episode_end')

    def test_drawdown_guard_liquidates(self):
        e=environment(max_drawdown=.00001,spread_bps=10.)
        _,_,done,_,info=e.step(0)
        self.assertTrue(done);self.assertEqual(info['done_reason'],'max_drawdown')
        self.assertEqual(e.position,0);self.assertEqual(len(e.trades_pnl),1)

    def test_time_exit_short_pays_ask(self):
        e=environment(spread_bps=10.,max_holding_bars=1)
        e.step(1);e.step(2)
        self.assertAlmostEqual(e.trades_meta[0]['exit_price'],10010)
        self.assertTrue(e.trades_meta[0]['hit_temps'])

    def test_gap_through_stop_fills_at_open(self):
        for action in (0,1):
            with self.subTest(action=action):
                e=environment();e.step(action)
                price=e.sl_price+(-50 if action==0 else 50)
                e.data.open[e.idx]=e.data.close[e.idx]=price
                e.data.low[e.idx]=e.data.high[e.idx]=price
                e.step(2)
                self.assertEqual(e.trades_meta[0]['exit_price'],price)

    def test_train_full_scaler_never_sees_validation(self):
        frame=pd.DataFrame({c:np.r_[np.arange(70),np.full(30,1e6)] for c in t.FEATURE_COLS})
        with patch.object(t,'load_mt5_data',return_value=frame), \
             patch.object(t,'create_datasets',return_value=(None,None,None,None)), \
             patch.object(t,'run_training_on_split') as run:
            t.run_training_full(t.PPOConfig())
        np.testing.assert_allclose(run.call_args.args[4]['mean'],34.5)

    def test_fold_scalers_use_only_each_training_slice(self):
        frame=pd.DataFrame({c:np.arange(100) for c in t.FEATURE_COLS})
        with patch.object(t,'load_mt5_data',return_value=frame), \
             patch.object(t,'create_datasets_from_slices',return_value=(None,None,None,None)), \
             patch.object(t,'run_training_on_split') as run:
            t.run_walkforward(t.PPOConfig(),.55,.15,.10,max_folds=3,auto_chain=False)
        self.assertEqual(run.call_count,3)
        for k,call in enumerate(run.call_args_list):
            np.testing.assert_allclose(call.args[4]['mean'],27.+10*k)

    def test_checkpoint_scaler_is_selected_by_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths=[]
            for k in (1,2):
                p=Path(tmp)/f'model{k}.pth';paths.append(str(p))
                np.savez(p.with_name(p.stem+'_norm.npz'),mean=np.full(len(t.FEATURE_COLS),k),
                         std=np.ones(len(t.FEATURE_COLS)),features=np.array(t.FEATURE_COLS))
                self.assertEqual(core.load_model_norm_stats(str(p))['mean'][0],k)
            with self.assertRaises(ValueError):core.load_shared_model_norm_stats(paths)


if __name__=='__main__':unittest.main()
