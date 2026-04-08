function status = build_carsim_simulink_template_v2(model_name, save_path)
% BUILD_CARSIM_SIMULINK_TEMPLATE_V2 生成并保存 V2 版本的 Simulink 模板模型
% 特点：使用 Bus Element In/Out 接口

try
    status = 0;
    
    %% ========= 0. 初始化 =========
    load_system('simulink');
    
    if bdIsLoaded(model_name)
        close_system(model_name, 0);
    end
    new_system(model_name);

    %% ========= 1. 定义 Bus 对象 =========
    clear elems elems_in;

    % ---- Carsim输出总线 ----
    elems(1)=Simulink.BusElement; elems(1).Name='Vx';
    elems(2)=Simulink.BusElement; elems(2).Name='Vy';
    elems(3)=Simulink.BusElement; elems(3).Name='Yaw_Rate';
    elems(4)=Simulink.BusElement; elems(4).Name='X_O';
    elems(5)=Simulink.BusElement; elems(5).Name='Y_O';

    Bus_CarSim_Out = Simulink.Bus;
    Bus_CarSim_Out.Elements = elems;
    assignin('base','Bus_CarSim_Out',Bus_CarSim_Out);

    % ---- Carsim输入总线 ----
    elems_in(1)=Simulink.BusElement; elems_in(1).Name='Steer_SW';
    elems_in(2)=Simulink.BusElement; elems_in(2).Name='Pbk_Con';

    Bus_CarSim_In = Simulink.Bus;
    Bus_CarSim_In.Elements = elems_in;
    assignin('base','Bus_CarSim_In',Bus_CarSim_In);

    %% ========= 2. 求解器配置 =========
    set_param(model_name,...
        'SolverType','Fixed-step',...
        'Solver','ode3',...
        'FixedStep','0.001');

    %% ========= 3. Carsim S-function =========
    cs_block = [model_name '/Carsim'];
    add_block('simulink/User-Defined Functions/S-Function', cs_block,...
        'Position',[100 120 250 180]);
    set_param(cs_block, 'FunctionName', 'vsc_sf');
    set_param(cs_block, 'Parameters', 'simfile.sim'); 

    %% ========= 4. Bus Selector =========
    sel_block = [model_name '/Bus_Selector'];
    add_block('built-in/BusSelector', sel_block,...
        'OutputSignals','Vx,Vy,Yaw_Rate,X_O,Y_O',...
        'Position',[320 120 330 200]);

    %% ========= 5. Controller 子系统 (使用 Bus Element) =========
    ctrl_block = [model_name '/Controller_Unit'];
    add_block('built-in/SubSystem', ctrl_block,...
        'TreatAsAtomicUnit','on',...
        'Position',[450 100 600 220]);

    % 进入子系统修改接口
    % 移除默认端口 (如果有)
    try, delete_block([ctrl_block '/In1']); catch, end
    try, delete_block([ctrl_block '/Out1']); catch, end

    % 使用 Bus Element In
    add_block('simulink/Signal Routing/Bus Element In', [ctrl_block '/State_In'], ...
        'Element', 'Vx', ...
        'Position', [30 80 60 110]);
    
    % 添加 MATLAB Function 核心
    add_block('simulink/User-Defined Functions/MATLAB Function', [ctrl_block '/Core'], ...
        'Position', [120 70 250 130]);

    % 使用 Bus Element Out
    add_block('simulink/Signal Routing/Bus Element Out', [ctrl_block '/Control_Out'], ...
        'Element', 'Steer_SW', ...
        'Position', [300 80 330 110]);

    % 子系统内部连线
    add_line(ctrl_block, 'State_In/1', 'Core/1');
    add_line(ctrl_block, 'Core/1', 'Control_Out/1');

    %% ========= 6. Bus Creator =========
    cre_block = [model_name '/Bus_Creator'];
    add_block('built-in/BusCreator', cre_block,...
        'Inputs','2',...
        'OutDataTypeStr','Bus: Bus_CarSim_In',...
        'NonVirtualBus','on',...
        'Position',[700 120 710 200]);

    %% ========= 7. 顶层连接 =========
    add_line(model_name, 'Carsim/1', 'Bus_Selector/1');
    add_line(model_name, 'Bus_Selector/1', 'Controller_Unit/1');
    add_line(model_name, 'Controller_Unit/1', 'Bus_Creator/1');
    add_line(model_name, 'Bus_Creator/1', 'Carsim/1');

    %% ========= 8. 保存 =========
    save_system(model_name, save_path);
    close_system(model_name, 0);

    status = 1;
    disp(['V2 model ' model_name ' successfully created']);

catch me
    status = 0;
    disp(['V2 build failed: ' me.message]);
    if bdIsLoaded(model_name)
        close_system(model_name, 0);
    end
end

end
