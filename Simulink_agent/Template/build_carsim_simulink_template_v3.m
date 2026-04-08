function status = build_carsim_simulink_template_v3(model_name, save_path)
% BUILD_CARSIM_SIMULINK_TEMPLATE_V3 生成并保存 V3 版本的 Simulink 模板模型
%
% 输入参数:
%   model_name: 模型名称
%   save_path:  保存路径
%
% 输出参数:
%   status:     运行状态 (1 为成功, 0 为失败)

try
    status = 0;
    
    %% ========= 0. 初始化 =========
    load_system('simulink');
    
    if bdIsLoaded(model_name)
        close_system(model_name, 0);
    end
    new_system(model_name);
    % open_system(model_name); % 如果不需要弹出窗口可注释

    %% ========= 1 Bus定义 =========
    clear elems elems_in;

    % ---- Carsim输出 ----
    elems(1)=Simulink.BusElement; elems(1).Name='Vx';
    elems(2)=Simulink.BusElement; elems(2).Name='Vy';
    elems(3)=Simulink.BusElement; elems(3).Name='Yaw_Rate';

    Bus_State = Simulink.Bus;
    Bus_State.Elements = elems;
    assignin('base','Bus_State',Bus_State);

    % ---- Carsim输入 ----
    elems_in(1)=Simulink.BusElement; elems_in(1).Name='Steer_SW';
    elems_in(2)=Simulink.BusElement; elems_in(2).Name='Pbk_Con';

    Bus_Control = Simulink.Bus;
    Bus_Control.Elements = elems_in;
    assignin('base','Bus_Control',Bus_Control);

    %% ========= 2 求解器 =========
    set_param(model_name,...
        'SolverType','Fixed-step',...
        'Solver','ode3',...
        'FixedStep','0.001');

    %% ========= 3 Carsim S-function =========
    add_block('simulink/User-Defined Functions/S-Function', ...
        [model_name '/Carsim'], ...
        'Position',[600 100 750 250]);
    
    set_param([model_name '/Carsim'], ...
        'FunctionName','vs_sf', ...
        'Parameters','simfile.sim');

    %% ========= 4 Bus Selector =========
    sel = [model_name '/BusSel'];
    add_block('built-in/BusSelector', sel,...
        'OutputSignals','Vx,Vy,Yaw_Rate',...
        'Position',[300 120 350 200]);

    %% ========= 5 状态Bus封装 =========
    stateBus = [model_name '/StateBusCreator'];
    add_block('built-in/BusCreator', stateBus,...
        'Inputs','3',...
        'OutDataTypeStr','Bus: Bus_State',...
        'NonVirtualBus','on',...
        'Position',[380 120 430 200]);

    %% ========= 6 Controller =========
    ctrl = [model_name '/Controller'];
    add_block('built-in/Subsystem', ctrl,...
        'TreatAsAtomicUnit','on',...
        'Position',[500 100 650 220]);

    % open_system(ctrl);

    % 删除默认
    try, delete_block([ctrl '/In1']); catch, end
    try, delete_block([ctrl '/Out1']); catch, end

    % Inport
    add_block('simulink/Sources/In1',...
        [ctrl '/state_in'],...
        'Position',[30 80 60 110]);

    % MATLAB Function
    add_block('simulink/User-Defined Functions/MATLAB Function',...
        [ctrl '/core'],...
        'Position',[120 70 250 130]);

    % 设置控制逻辑代码
    % try
    %     set_param([ctrl '/core'], 'MATLABFcn', sprintf([ ...
    %     'function u = fcn(Vx,Vy,Yaw_Rate)\n' ...
    %     'steer = -0.1 * Yaw_Rate;\n' ...
    %     'brake = 0;\n' ...
    %     'u = [steer; brake];\n' ...
    %     'end']));
    % catch
    %     disp('⚠️ Warning: Could not set MATLAB Function script via set_param.');
    % end

    % Outport
    add_block('simulink/Sinks/Out1',...
        [ctrl '/control_out'],...
        'Position',[300 80 330 110]);

    % 连线
    add_line(ctrl,'state_in/1','core/1');
    add_line(ctrl,'core/1','control_out/1');

    % close_system(ctrl);

    %% ========= 7 控制Bus =========
    ctrlBus = [model_name '/ControlBusCreator'];
    add_block('built-in/BusCreator', ctrlBus,...
        'Inputs','2',...
        'OutDataTypeStr','Bus: Bus_Control',...
        'NonVirtualBus','on',...
        'Position',[700 120 750 200]);

    %% ========= 8 连接 =========
    add_line(model_name,'Carsim/1','BusSel/1');
    add_line(model_name,'BusSel/1','StateBusCreator/1');
    add_line(model_name,'BusSel/2','StateBusCreator/2');
    add_line(model_name,'BusSel/3','StateBusCreator/3');

    add_line(model_name,'StateBusCreator/1','Controller/1');
    add_line(model_name,'Controller/1','ControlBusCreator/1');
    add_line(model_name,'ControlBusCreator/1','Carsim/1');

    %% ========= 9 保存 =========
    save_system(model_name, save_path);
    close_system(model_name, 0);

    status = 1;
    disp(['V3 model ' model_name ' successfully created']);

catch me
    status = 0;
    disp(['V3 build failed: ' me.message]);
    if bdIsLoaded(model_name)
        close_system(model_name, 0);
    end
end

end
