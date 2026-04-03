function build_carsim_template()

%% ========= 0. 初始化 =========
load_system('simulink'); % 预加载核心库
model_name = 'carsim_template_auto';
new_system(model_name);
% open_system(model_name);

%% ========= 1. 定义 Bus =========
% -------- StateBus --------
clear elems; % 确保 elems 没被预定义为 double

elems(1) = Simulink.BusElement;
elems(1).Name = 'vx';

elems(2) = Simulink.BusElement;
elems(2).Name = 'vy';

elems(3) = Simulink.BusElement;
elems(3).Name = 'yaw_rate';

StateBus = Simulink.Bus;
StateBus.Elements = elems;

assignin('base','StateBus',StateBus);

% -------- ControlBus --------
clear elems2; % 确保 elems2 没被预定义为 double

elems2(1) = Simulink.BusElement;
elems2(1).Name = 'steering';

elems2(2) = Simulink.BusElement;
elems2(2).Name = 'throttle';

elems2(3) = Simulink.BusElement;
elems2(3).Name = 'brake';

ControlBus = Simulink.Bus;
ControlBus.Elements = elems2;

assignin('base','ControlBus',ControlBus);

%% ========= 2. 添加 Controller 子系统 =========
add_block('simulink/Ports & Subsystems/Subsystem', ...
    [model_name '/Controller'], ...
    'Position',[100 100 300 250]);

% open_system([model_name '/Controller']);

% 删除默认内容
delete_line([model_name '/Controller'],'In1/1','Out1/1');
delete_block([model_name '/Controller/In1']);
delete_block([model_name '/Controller/Out1']);

% 添加 Bus 输入
add_block('simulink/Sources/In1', ...
    [model_name '/Controller/state_bus'], ...
    'Position',[50 80 80 120]);

% set_param([model_name '/Controller/state_bus'], 'PortName','state');

% 添加 MATLAB Function 控制器
add_block('simulink/User-Defined Functions/MATLAB Function', ...
    [model_name '/Controller/controller_core'], ...
    'Position',[120 70 250 130]);

% 注释掉设置代码的部分，避免版本或路径问题
% set_param([model_name '/Controller/controller_core'], ...
%     'MATLABFcn', sprintf([ ...
%     'function u = fcn(vx,vy,yaw_rate)\n' ...
%     '%% 简单控制器示例（可替换MPC）\n' ...
%     'steering = -0.1 * yaw_rate;\n' ...
%     'throttle = 0.2;\n' ...
%     'brake = 0;\n' ...
%     'u = [steering; throttle; brake];\n' ...
%     'end']));

% 添加 Bus 输出
add_block('simulink/Sinks/Out1', ...
    [model_name '/Controller/control_bus'], ...
    'Position',[300 80 330 120]);

% set_param([model_name '/Controller/control_bus'], 'PortName','control');

% 连接（注意：需要后续手动或脚本扩展为多端口）
add_line([model_name '/Controller'], 'state_bus/1', 'controller_core/1');
add_line([model_name '/Controller'], 'controller_core/1', 'control_bus/1');

%% ========= 3. Interface Layer =========
add_block('simulink/Ports & Subsystems/Subsystem', ...
    [model_name '/Interface'], ...
    'Position',[350 100 550 250]);

%% ========= 4. Carsim S-function =========
add_block('simulink/User-Defined Functions/S-Function', ...
    [model_name '/Carsim'], ...
    'Position',[600 100 750 250]);

% ⚠️ 根据你的Carsim版本修改
set_param([model_name '/Carsim'], ...
    'FunctionName','vs_sf', ...
    'Parameters','simfile.sim');

%% ========= 5. Bus Creator（控制输入） =========
add_block('simulink/Signal Routing/Bus Creator', ...
    [model_name '/ControlBusCreator'], ...
    'Position',[280 150 320 200]);

set_param([model_name '/ControlBusCreator'], ...
    'Inputs','3');

%% ========= 6. Bus Creator（状态输出） =========
add_block('simulink/Signal Routing/Bus Creator', ...
    [model_name '/StateBusCreator'], ...
    'Position',[800 150 840 200]);

set_param([model_name '/StateBusCreator'], ...
    'Inputs','3');

%% ========= 7. To Workspace =========
add_block('simulink/Sinks/To Workspace', ...
    [model_name '/ToWorkspace'], ...
    'Position',[900 150 1000 200]);

set_param([model_name '/ToWorkspace'], ...
    'VariableName','simout', ...
    'SaveFormat','StructureWithTime');

%% ========= 8. 顶层连接 =========
add_line(model_name,'Controller/1','ControlBusCreator/1');
add_line(model_name,'ControlBusCreator/1','Carsim/1');

add_line(model_name,'Carsim/1','StateBusCreator/1');
add_line(model_name,'StateBusCreator/1','ToWorkspace/1');

add_line(model_name,'StateBusCreator/1','Controller/1');

%% ========= 9. 求解器配置 =========
set_param(model_name, ...
    'SolverType','Fixed-step', ...
    'Solver','ode4', ...
    'FixedStep','0.01', ...
    'StopTime','10');

%% ========= 10. 保存模型 =========
save_system(model_name);

disp('✅ Simulink模板已生成完成！');

end