
torch-jit-export]
0
1129 "Conv*
	dilations00*	
group*
kernels00*
pads0000*
strides00Q
129
2
3
4
5130 "	SpatialBN*
epsilon��'7*
is_test*
momentum  �?
130131 "Relu_
131
7132 "Conv*
	dilations00*	
group*
kernels00*
pads0000*
strides00S
132
8
9
10
11133 "	SpatialBN*
epsilon��'7*
is_test*
momentum  �?
133134 "Relu`
134
13135 "Conv*
	dilations00*	
group*
kernels00*
pads0000*
strides00U
135
14
15
16
17136 "	SpatialBN*
epsilon��'7*
is_test*
momentum  �?
136
131137 "Add
137138 "Relu`
138
19139 "Conv*
	dilations00*	
group*
kernels00*
pads0000*
strides00U
139
20
21
22
23140 "	SpatialBN*
epsilon��'7*
is_test*
momentum  �?
140141 "Relu`
141
25142 "Conv*
	dilations00*	
group*
kernels00*
pads0000*
strides00U
142
26
27
28
29143 "	SpatialBN*
epsilon��'7*
is_test*
momentum  �?
143
138144 "Add
144145 "Relu`
145
31146 "Conv*
	dilations00*	
group*
kernels00*
pads0000*
strides00U
146
32
33
34
35147 "	SpatialBN*
epsilon��'7*
is_test*
momentum  �?
147148 "Relu`
148
37149 "Conv*
	dilations00*	
group*
kernels00*
pads0000*
strides00U
149
38
39
40
41150 "	SpatialBN*
epsilon��'7*
is_test*
momentum  �?
150
145151 "Add
151152 "Relu`
152
43153 "Conv*
	dilations00*	
group*
kernels00*
pads0000*
strides00U
153
44
45
46
47154 "	SpatialBN*
epsilon��'7*
is_test*
momentum  �?
154155 "Relu`
155
49156 "Conv*
	dilations00*	
group*
kernels00*
pads0000*
strides00U
156
50
51
52
53157 "	SpatialBN*
epsilon��'7*
is_test*
momentum  �?`
152
55158 "Conv*
	dilations00*	
group*
kernels00*
pads0 0 0 0 *
strides00U
158
56
57
58
59159 "	SpatialBN*
epsilon��'7*
is_test*
momentum  �?
157
159160 "Add
160161 "Relu`
161
61162 "Conv*
	dilations00*	
group*
kernels00*
pads0000*
strides00U
162
62
63
64
65163 "	SpatialBN*
epsilon��'7*
is_test*
momentum  �?
163164 "Relu`
164
67165 "Conv*
	dilations00*	
group*
kernels00*
pads0000*
strides00U
165
68
69
70
71166 "	SpatialBN*
epsilon��'7*
is_test*
momentum  �?
166
161167 "Add
167168 "Relu`
168
73169 "Conv*
	dilations00*	
group*
kernels00*
pads0000*
strides00U
169
74
75
76
77170 "	SpatialBN*
epsilon��'7*
is_test*
momentum  �?
170171 "Relu`
171
79172 "Conv*
	dilations00*	
group*
kernels00*
pads0000*
strides00U
172
80
81
82
83173 "	SpatialBN*
epsilon��'7*
is_test*
momentum  �?
173
168174 "Add
174175 "Relu`
175
85176 "Conv*
	dilations00*	
group*
kernels00*
pads0000*
strides00U
176
86
87
88
89177 "	SpatialBN*
epsilon��'7*
is_test*
momentum  �?
177178 "Relu`
178
91179 "Conv*
	dilations00*	
group*
kernels00*
pads0000*
strides00U
179
92
93
94
95180 "	SpatialBN*
epsilon��'7*
is_test*
momentum  �?`
175
97181 "Conv*
	dilations00*	
group*
kernels00*
pads0 0 0 0 *
strides00W
181
98
99
100
101182 "	SpatialBN*
epsilon��'7*
is_test*
momentum  �?
180
182183 "Add
183184 "Relua
184
103185 "Conv*
	dilations00*	
group*
kernels00*
pads0000*
strides00Y
185
104
105
106
107186 "	SpatialBN*
epsilon��'7*
is_test*
momentum  �?
186187 "Relua
187
109188 "Conv*
	dilations00*	
group*
kernels00*
pads0000*
strides00Y
188
110
111
112
113189 "	SpatialBN*
epsilon��'7*
is_test*
momentum  �?
189
184190 "Add
190191 "Relua
191
115192 "Conv*
	dilations00*	
group*
kernels00*
pads0000*
strides00Y
192
116
117
118
119193 "	SpatialBN*
epsilon��'7*
is_test*
momentum  �?
193194 "Relua
194
121195 "Conv*
	dilations00*	
group*
kernels00*
pads0000*
strides00Y
195
122
123
124
125196 "	SpatialBN*
epsilon��'7*
is_test*
momentum  �?
196
191197 "Add
197198 "ReluF
198199 "PadImage*
mode"constant*
pads0 0 0 0 *
value    G
199200 "AveragePool*
kernels00*
pads0 0 0 0 *
strides00
200201 "Shape
201OC2_DUMMY_0 "Shape:OC2_DUMMY_1 "GivenTensorIntFill*	
shape0*

values0 <OC2_DUMMY_2 "GivenTensorInt64Fill*	
shape0*

values0 @
OC2_DUMMY_0OC2_DUMMY_3 "ConstantFill*	
dtype
*	
value E
OC2_DUMMY_3
OC2_DUMMY_1
OC2_DUMMY_2OC2_DUMMY_3 "ScatterAssign*
OC2_DUMMY_3OC2_DUMMY_4 "Cast*
to<OC2_DUMMY_5 "GivenTensorInt64Fill*	
shape0*

values0I
OC2_DUMMY_0OC2_DUMMY_6 "ConstantFill*	
dtype
*
value���������E
OC2_DUMMY_6
OC2_DUMMY_1
OC2_DUMMY_5OC2_DUMMY_6 "ScatterAssign*
OC2_DUMMY_6OC2_DUMMY_7 "Cast*
to-
201
OC2_DUMMY_4
OC2_DUMMY_7202 "Slice
202203 "Squeeze*
dims0 9204"GivenTensorInt64Fill*
values0���������*
shape"
203205 "
ExpandDims*
axes0 "
204206 "
ExpandDims*
axes0 0
205
206207OC2_DUMMY_8 "Concat*
axis '
200
207208OC2_DUMMY_9 "Reshape
208
127
128209 "FC*  :0:1:2:3:4:5:6:7:8:9:10:11:12:13:14:15:16:17:18:19:20:21:22:23:24:25:26:27:28:29:30:31:32:33:34:35:36:37:38:39:40:41:42:43:44:45:46:47:48:49:50:51:52:53:54:55:56:57:58:59:60:61:62:63:64:65:66:67:68:69:70:71:72:73:74:75:76:77:78:79:80:81:82:83:84:85:86:87:88:89:90:91:92:93:94:95:96:97:98:99:100:101:102:103:104:105:106:107:108:109:110:111:112:113:114:115:116:117:118:119:120:121:122:123:124:125:126:127:128B209