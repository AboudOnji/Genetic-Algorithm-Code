% --- Script de MATLAB para visualizar la función objetivo f(x1,x2,x3,x4) ---

% Definir la función objetivo
objectiveFunc = @(x1, x2, x3, x4) ...
    (10 * (x2 - x1.^2)).^2 + ...
    (1 - x1).^2 + ...
    90 * (x4 - x3.^2).^2 + ... % Corregido según la fórmula: 90 * (diff)^2
    (1 - x3).^2 + ...
    10 * (x2 + x4 - 2).^2 + ...
    0.1 * (x2 - x4).^2;

% --- Visualización 1: Variando x1 y x2, con x3 y x4 fijos en su óptimo (1) ---
disp('Generando gráfico para x1, x2 (con x3=1, x4=1)...');

% Valores fijos para x3 y x4 (óptimos)
x3_fixed_opt = 1;
x4_fixed_opt = 1;

% Rango para x1 y x2. El óptimo es x1=1, x2=1.
% Exploremos un área alrededor del óptimo y un poco más amplio.
x1_range = linspace(-2, 3, 100); % 100 puntos entre -2 y 3
x2_range = linspace(-1, 4, 100); % 100 puntos entre -1 y 4

% Crear una malla de puntos para x1 y x2
[X1, X2] = meshgrid(x1_range, x2_range);

% Calcular los valores de la función F para cada punto en la malla
F_slice1 = objectiveFunc(X1, X2, x3_fixed_opt, x4_fixed_opt);

% --- Gráfico de Superficie (Surface Plot) ---
figure; % Crear una nueva figura
surf(X1, X2, F_slice1);
xlabel('x1');
ylabel('x2');
zlabel('f(x1, x2, x3=1, x4=1)');
title('Función Objetivo (Rebanada con x3=1, x4=1)');
colorbar; % Muestra la barra de colores para los valores de F
shading interp; % Suaviza los colores de la superficie (opcional)
view(3); % Vista 3D estándar
rotate3d on; % Permite rotar el gráfico con el ratón
% Guardar la figura (opcional)
% print('funcion_objetivo_slice_x1x2_opt.png', '-dpng');

% --- Gráfico de Contorno (Contour Plot) ---
figure; % Crear otra nueva figura
contour(X1, X2, F_slice1, 50); % 50 niveles de contorno
xlabel('x1');
ylabel('x2');
title('Contornos de f(x1, x2, x3=1, x4=1)');
colorbar;
axis equal;
hold on;
plot(1, 1, 'r*', 'MarkerSize', 10, 'LineWidth', 2); % Marcar el óptimo (1,1) para x1,x2
legend('Niveles de f', 'Óptimo (1,1)');
hold off;
% Guardar la figura (opcional)
% print('funcion_objetivo_contorno_x1x2_opt.png', '-dpng');


% --- Visualización 2: Variando x3 y x4, con x1 y x2 fijos en su óptimo (1) ---
disp('Generando gráfico para x3, x4 (con x1=1, x2=1)...');

% Valores fijos para x1 y x2 (óptimos)
x1_fixed_opt = 1;
x2_fixed_opt = 1;

% Rango para x3 y x4
x3_range = linspace(-2, 3, 100);
x4_range = linspace(-1, 4, 100);

[X3, X4] = meshgrid(x3_range, x4_range);
F_slice2 = objectiveFunc(x1_fixed_opt, x2_fixed_opt, X3, X4);

% --- Gráfico de Superficie (Surface Plot) ---
figure;
surf(X3, X4, F_slice2);
xlabel('x3');
ylabel('x4');
zlabel('f(x1=1, x2=1, x3, x4)');
title('Función Objetivo (Rebanada con x1=1, x2=1)');
colorbar;
shading interp;
view(3);
rotate3d on;

% --- Gráfico de Contorno (Contour Plot) ---
figure;
contour(X3, X4, F_slice2, 50);
xlabel('x3');
ylabel('x4');
title('Contornos de f(x1=1, x2=1, x3, x4)');
colorbar;
axis equal;
hold on;
plot(1, 1, 'r*', 'MarkerSize', 10, 'LineWidth', 2); % Marcar el óptimo (1,1) para x3,x4
legend('Niveles de f', 'Óptimo (1,1)');
hold off;

disp('Visualizaciones generadas.');
disp('Puedes modificar los rangos (ej. x1_range) o los valores fijos (ej. x3_fixed_opt) para explorar otras partes de la función.');
% optimizacion usando GA
% Definir los límites de las variables
lb = [-20, -20, -20, -20]; % Límites inferiores
ub = [20, 20, 20, 20]; % Límites superiores
options= optimoptions('ga', 'Display','diagnose','EliteCount', 2, ...
    'PopulationSize', 100, 'MaxGenerations', 200, ...
    'FunctionTolerance', 1e-6,'CrossoverFcn','crossoversinglepoint','MutationFcn','mutationuniform','PlotFcn','gaplotbestf');
[x, fval] = ga(@(x)objectiveFunc(x(1), x(2), x(3), x(4)), 4, [], [], [], [], lb, ub, [], options);