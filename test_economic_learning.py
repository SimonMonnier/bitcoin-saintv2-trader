import unittest
from unittest.mock import patch
import training as t
import numpy as np
import pandas as pd
import torch
from economic_learning import downside_score,causal_features,net_trade_targets,simulate_entries
from research_models import TemporalPatchRegressor


class NetLearningTests(unittest.TestCase):
    def frame(self):
        rng=np.random.default_rng(5)
        close=10000+np.cumsum(rng.normal(0,12,800))
        op=np.r_[close[0],close[:-1]]
        return pd.DataFrame({'open':op,'close':close,'high':np.maximum(op,close)+20,
                'low':np.minimum(op,close)-20,'atr_14':np.full(800,25.)})

    def test_downside_constant_losses_finite(self):
        self.assertAlmostEqual(downside_score([-1]*50),-1.)
        self.assertAlmostEqual(downside_score([0,0,0]),0.)
        self.assertAlmostEqual(downside_score([-1,1]),0.)

    def test_features_do_not_use_entry_candle_or_future(self):
        f=self.frame();first=causal_features(f,['close'],True)
        f.loc[500:,'close']=9999999
        second=causal_features(f,['close'],True)
        pd.testing.assert_frame_equal(first.iloc[:501],second.iloc[:501])

    def test_target_matches_environment_with_deterministic_costs(self):
        f=self.frame()
        for c in t.FEATURE_COLS:
            if c not in f:f[c]=0.
        cfg=t.PPOConfig(episode_length=300,max_holding_bars=20,use_vol_curriculum=False,
                        tick_noise_bps=0,use_be_trail=False,max_drawdown=.99,min_capital_frac=0.)
        indices=np.array([100,160,200,250,350])
        targets,exits=net_trade_targets(f,indices,cfg)
        data=t.MarketData(f,t.FEATURE_COLS)
        for k,i in enumerate(indices):
            for action in (0,1):
                env=t.BTCTradingEnvDiscrete(data,cfg)
                env.idx=env.start_idx=int(i);env.end_idx=int(i)+100
                env._sample_trade_spread_bps=lambda:cfg.spread_bps
                with patch.object(t.np.random,'uniform',side_effect=lambda a,b:(a+b)/2):
                    env.step(action)
                    while env.position:env.step(2)
                self.assertAlmostEqual(env.trades_pnl[0]/env.risk_amount,targets[k,action],places=3)
                self.assertEqual(env.trades_meta[0]['exit_idx'],exits[k,action])

    def test_unresolved_trade_kept_with_exit_cost(self):
        f=pd.DataFrame({c:np.full(800,10000.) for c in ('open','close','high','low')})
        f['atr_14']=100
        cfg=t.PPOConfig(max_holding_bars=10)
        y,end=net_trade_targets(f,[100],cfg)
        self.assertTrue((y<0).all());np.testing.assert_array_equal(end,[[110,110]])

    def test_split_boundary_rejected(self):
        with self.assertRaises(ValueError):net_trade_targets(self.frame(),[799],t.PPOConfig())

    def test_simulation_has_no_overlapping_positions(self):
        scores=np.ones((4,2));ys=np.ones((4,2));ends=np.array([[3,3],[4,4],[5,5],[6,6]])
        result=simulate_entries([1,2,3,4],scores,ys,ends)
        self.assertEqual(result['entries'],[1,4])

    def test_patch_prediction_is_batch_independent(self):
        torch.manual_seed(0)
        m=TemporalPatchRegressor(3).eval();x=torch.randn(2,96,3)
        with torch.no_grad():
            np.testing.assert_allclose(m(x[:1]).numpy(),m(x)[:1].numpy(),atol=1e-6)

if __name__=='__main__':unittest.main()
