import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from .models import Trans_Encoder


class BilateralLocalityContextAttention(nn.Module):

    def __init__(self, in_channels, inter_channels):
        super(BilateralLocalityContextAttention, self).__init__()

        self.w1 = nn.Linear(256, 256)
        self.w2 = nn.Linear(256, 256)
        self.v_fc = nn.Linear(256, 256)
        self.a_fc = nn.Linear(256, 256)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1) # default=0.1
        self.layer_norm = nn.LayerNorm(256, eps=1e-6)

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # conv+bn+relu的顺序
        self.conv_1 = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                        kernel_size=(3,3), stride=1, padding=(1,1)),
                nn.BatchNorm2d(self.inter_channels),
                nn.ReLU()
            )

        self.conv_2 = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels, out_channels=self.inter_channels,
                        kernel_size=(1,3), stride=1, padding=(0,1)),
                nn.BatchNorm2d(self.inter_channels),
                nn.ReLU()
            )

        self.conv_3 = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels, out_channels=self.inter_channels,
                        kernel_size=(3,1), stride=1, padding=(1,0)),
                nn.BatchNorm2d(self.inter_channels),
                nn.ReLU()
            )

        self.conv_4 = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels, out_channels=2,
                        kernel_size=(1,1), stride=1, padding=0),
                nn.ReLU()
            )

    def forward(self, v, a):
        # x [bz,T,dim]
        # y [bz,T,dim]
        v1 = self.w1(v)
        a1 = self.w1(a)
        sim = torch.matmul(v1, a1.permute(0, 2, 1)) # [bz,T,T]

        x = self.conv_1(sim.unsqueeze(1))
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)

        sim_v_a = sim + x[:,0,:,:].squeeze()
        sim_a_v = (sim + x[:,1,:,:].squeeze()).transpose(2, 1)

        att_v_a = torch.softmax(sim_v_a, dim=-1)
        att_a_v = torch.softmax(sim_a_v, dim=-1)
        atted_a = torch.matmul(att_v_a, a)
        atted_v = torch.matmul(att_a_v, v)

        v = self.dropout(self.relu(self.v_fc(v + atted_a)))
        a = self.dropout(self.relu(self.a_fc(a + atted_v)))
        v = self.layer_norm(v)
        a = self.layer_norm(a)

        return v, a

class AudioAdaptiveFilter(nn.Module):
    def __init__(self):
        super(AudioAdaptiveFilter, self).__init__()
        self.hidden_size = 512
        self.relu = nn.ReLU()

        self.sigmoid = nn.Sigmoid()

        # spatial attention
        self.affine_video_2 = nn.Linear(256, 128)
        self.affine_audio_2 = nn.Linear(256, 128)
        self.affine_v_s_att = nn.Linear(256, 1)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, visual_feature, audio):
        '''
        :param visual_feature: [batch, 10, 36, 512]
        :param audio_feature:  [batch, 10, 128]
        :return: [batch, 10, 512]
        '''
        audio = audio.transpose(1, 0)
        batch, t_size, o, v_dim = visual_feature.size()
        a_dim = audio.size(-1)
        audio_feature = audio.reshape(batch * t_size, a_dim)

        c_att_visual_feat = visual_feature.reshape(batch * t_size, -1, v_dim)
        c_att_visual_query = self.relu(self.affine_video_2(c_att_visual_feat)) # [640, 36, 128]
        audio_query_2 = self.relu(self.affine_audio_2(audio_feature)).unsqueeze(-2) # [640, 1, 128]
        audio_query_2 = audio_query_2.repeat(1, o, 1)
        audio_video_query_2 = torch.cat([c_att_visual_query, audio_query_2], dim=-1)
        spatial_att_maps = self.affine_v_s_att(audio_video_query_2).squeeze()

        spatial_att_maps = self.tanh(spatial_att_maps)

        mask = (spatial_att_maps < 0)
        spatial_att_maps = spatial_att_maps.masked_fill(mask, -1e9)

        spatial_att_maps = torch.softmax(spatial_att_maps.unsqueeze(1), dim=-1)

        c_s_att_visual_feat = torch.bmm(spatial_att_maps, c_att_visual_feat).squeeze().reshape(batch, t_size, v_dim)

        return c_s_att_visual_feat


class supv_main_model(nn.Module):
    def __init__(self):
        super(supv_main_model, self).__init__()

        self.video_input_dim = 1536
        self.d_model = 256

        self.spatial_att = AudioAdaptiveFilter()

        self.audio_fc = nn.Sequential(nn.Linear(2048, 256),
                                      nn.LeakyReLU(),
                                      )

        # 576, 144, 36
        # 384, 768, 1536
        self.visual_fc = nn.Sequential(nn.Linear(self.video_input_dim, 256),
                                      nn.LeakyReLU(),
                                       )

        self.temporalencoder = Trans_Encoder(d_model=256, num_layers=2, nhead=4)

        self.loc_win_att = BilateralLocalityContextAttention(1, 32)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

        self.v_fc = nn.Linear(256, 256, bias=False)
        self.a_fc = nn.Linear(256, 256, bias=False)
        self.layer_norm = nn.LayerNorm(256, eps=1e-6)

        self.predict_fc = nn.Linear(256, 29)


    def forward(self, visual_feat, audio):

        visual_feature = self.visual_fc(visual_feat)
        audio = self.audio_fc(audio)

        visual_feature = self.spatial_att(visual_feature, audio)

        visual_feature = visual_feature.transpose(1, 0).contiguous()
        audio = audio.transpose(1, 0).contiguous()

        visual_feature, audio = self.temporalencoder(visual_feature, audio)
        visual_feature = visual_feature.transpose(1, 0).contiguous()
        audio = audio.transpose(1, 0).contiguous()
        visual_feature, audio = self.loc_win_att(visual_feature, audio)
        visual_feature = visual_feature.transpose(1, 0).contiguous()
        audio = audio.transpose(1, 0).contiguous()

        visual = self.dropout(self.relu(self.v_fc(visual_feature)))
        audio = self.dropout(self.relu(self.a_fc(audio)))
        visual = self.layer_norm(visual)
        audio = self.layer_norm(audio)

        video_query_output = torch.mul(visual + audio, 0.5)

        scores = self.predict_fc(video_query_output.transpose(1, 0)) # [bz, 10, 29]

        scores = F.softmax(scores, dim=-1)

        return scores, visual.transpose(1, 0), audio.transpose(1, 0)




