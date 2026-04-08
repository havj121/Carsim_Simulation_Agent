function status = build_carsim_simulink_template_v1(model_name, save_path)
% BUILD_CARSIM_SIMULINK_TEMPLATE_V1 生成并保存指定名称和路径的 Simulink 模板模型
%
% 输入参数:
%   model_name: 模型名称 (例如 'carsim_template')
%   save_path:  保存路径 (包含 .slx 后缀, 例如 'D:/.../carsim_template.slx')
%
% 输出参数:
%   status:     运行状态 (1 为成功, 0 为失败)

try
    status = 0; % 默认失败状态
    
    %% ========= 0. 初始化 =========
    load_system('simulink'); % 预加载核心库
    
    % 如果模型已经打开，先关闭
    if bdIsLoaded(model_name)
        close_system(model_name, 0);
    end
    
    % 创建新模型
    new_system(model_name);
    
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
    
    % 删除默认内容
    delete_line([model_name '/Controller'],'In1/1','Out1/1');
    delete_block([model_name '/Controller/In1']);
    delete_block([model_name '/Controller/Out1']);
    
    % 添加 Bus 输入
    add_block('simulink/Sources/In1', ...
        [model_name '/Controller/state_bus'], ...
        'Position',[50 80 80 120]);
    
    % 添加 MATLAB Function 控制器
    add_block('simulink/User-Defined Functions/MATLAB Function', ...
        [model_name '/Controller/controller_core'], ...
        'Position',[120 70 250 130]);
    
    % 添加 Bus 输出
    add_block('simulink/Sinks/Out1', ...
        [model_name '/Controller/control_bus'], ...
        'Position',[300 80 330 120]);
    
    % 连接
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
    save_system(model_name, save_path);
    close_system(model_name, 0);
    
    status = 1; % 执行成功
    disp(['Model ' model_name ' successfully created and saved to: ' save_path]);
    
catch me
    status = 0; % 执行失败
    disp(['Failed to create model: ' me.message]);
    if bdIsLoaded(model_name)
        close_system(model_name, 0);
    end
end

end
