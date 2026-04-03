%% 1. 定义总线对象 (Bus Objects)
% 定义从 CarSim 输出到 Controller 的信号总线
clear Bus_CarSim_Out;
elems(1) = Simulink.BusElement;
elems(1).Name = 'Vx'; % 纵向速度
elems(2) = Simulink.BusElement;
elems(2).Name = 'Vy'; % 横向速度
elems(3) = Simulink.BusElement;
elems(3).Name = 'Yaw_Rate'; % 横摆角速度
elems(4) = Simulink.BusElement;
elems(4).Name = 'X_O'; % 全局坐标X
elems(5) = Simulink.BusElement;
elems(5).Name = 'Y_O'; % 全局坐标Y

Bus_CarSim_Out = Simulink.Bus;
Bus_CarSim_Out.Elements = elems;

% 定义从 Controller 输入到 CarSim 的信号总线
clear Bus_CarSim_In;
elems_in(1) = Simulink.BusElement;
elems_in(1).Name = 'Steer_SW'; % 方向盘转角
elems_in(2) = Simulink.BusElement;
elems_in(2).Name = 'Pbk_Con'; % 制动压力

Bus_CarSim_In = Simulink.Bus;
Bus_CarSim_In.Elements = elems_in;

% 将总线对象保存到基础工作区
assignin('base', 'Bus_CarSim_Out', Bus_CarSim_Out);
assignin('base', 'Bus_CarSim_In', Bus_CarSim_In);

%% 2. 创建并配置模型
modelName = 'CarSim_Static_Template';
if bdIsLoaded(modelName), close_system(modelName, 0); end
new_system(modelName);
open_system(modelName);

% 设置求解器为固定步长 (适配 HiL)
set_param(modelName, 'SolverType', 'Fixed-step', 'Solver', 'ode3', 'FixedStep', '0.001');

%% 3. 添加模块
% 添加 CarSim S-Function (核心内核)
cs_block = [modelName, '/CarSim_SF'];
add_block('built-in/S-Function', cs_block);
set_param(cs_block, 'FunctionName', 'vsc_sf', 'Parameters', '''simfile.sim''', 'Position', [100, 100, 250, 160]);

% 添加 Bus Selector (解析 CarSim 输出)
sel_block = [modelName, '/Bus_Selector'];
add_block('built-in/BusSelector', sel_block);
set_param(sel_block, 'OutputSignals', 'Vx,Vy,Yaw_Rate,X_O,Y_O', 'Position', [320, 100, 330, 200]);

% 添加控制器子系统 (Atomic Subsystem)
ctrl_block = [modelName, '/Controller_Unit'];
add_block('built-in/SubSystem', ctrl_block);
set_param(ctrl_block, 'TreatAsAtomicUnit', 'on', 'Position', [450, 80, 600, 220]);

% 在子系统内部创建接口 (此处简化，仅创建 In1/Out1)
add_block('built-in/Inport', [ctrl_block, '/In1'], 'Position', [20, 20, 50, 35]);
add_block('built-in/Outport', [ctrl_block, '/Out1'], 'Position', [200, 20, 230, 35]);

% 添加 Bus Creator (整合控制命令回传给 CarSim)
cre_block = [modelName, '/Bus_Creator'];
add_block('built-in/BusCreator', cre_block);
set_param(cre_block, 'Inputs', '2', 'OutDataTypeStr', 'Bus: Bus_CarSim_In', 'NonVirtualBus', 'on', 'Position', [700, 100, 710, 200]);

%% 4. 自动连线
% CarSim 输出 -> Bus Selector
add_line(modelName, 'CarSim_SF/1', 'Bus_Selector/1', 'autoroute', 'on');

% Bus Selector 信号 -> Controller (示例连接第一个信号)
add_line(modelName, 'Bus_Selector/1', 'Controller_Unit/1', 'autoroute', 'on');

% Controller -> Bus Creator
add_line(modelName, 'Controller_Unit/1', 'Bus_Creator/1', 'autoroute', 'on');

% Bus Creator -> CarSim 输入 (闭环)
add_line(modelName, 'Bus_Creator/1', 'CarSim_SF/1', 'autoroute', 'on');

%% 5. 保存
save_system(modelName);
disp(['模型 ', modelName, ' 构建完成！']);