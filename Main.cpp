#include <iostream>
#include <vector>
#include <cstdint>
#include <fstream>
#include <Eigen/Dense>
#include <string>

#define LaGarge

using PointsVector = std::vector<std::tuple<double,double>>;

PointsVector LoadDataFromFile(const std::string& path)
{
    PointsVector Output;
    std::ifstream InputFile(path);

    if (!InputFile)
    {
        std::cerr << "Failed to open the file." << std::endl;
        return Output;
    }

    double x, y;
    while (InputFile >> x >> y)
    {
        Output.emplace_back(x, y);
    }

    return Output;
}

void ExportDataToFile(const PointsVector& data, std::string fileName)
{
    std::ofstream Out("2018_paths/" + fileName);
    for (auto [x, y] : data)
    {
        Out << x << " " << y << std::endl;
    }
}

double LaGrangeInterpolation(double x, const PointsVector& nodes)
{
    double interpolatedValue = 0.0;

    for (const auto& [xi, yi] : nodes)
    {
        double term = yi;

        for (const auto& [xj, _] : nodes)
        {
            if (xj != xi)
            {
                term *= (x - xj) / (xi - xj);
            }
        }

        interpolatedValue += term;
    }
    return interpolatedValue;
}

double SplineInterpolation(double x, const PointsVector& nodes, const std::vector<double>& coefficients) 
{
    for (int i = 0; i < nodes.size() - 1; i++)
    {
        if (x >= std::get<0>(nodes[i]) && x <= std::get<0>(nodes[i + 1]))
        {
            double y = 0.0;
            for (int k = 0; k < 4; k++) 
            {
                y += coefficients[4 * i + k] * std::pow(x - std::get<0>(nodes[i]), k);
            }
            return y;
        }
    }
    return 0.0;
}

void InitializeMatricies(Eigen::MatrixXd& A, Eigen::VectorXd& b, size_t N, const PointsVector& nodes)
{

    A(0, 0) = 1.0;
    b(0) = std::get<1>(nodes[0]);

    A(1, 2) = 2.0;
    b(1) = 0.0;

    double h = std::get<0>(nodes[1]) - std::get<0>(nodes[0]);

    A(1, 1) = A(1, 1) + 1.0;
    A(1, 2) = A(1, 2) + 2 * h;
    A(1, 3) = A(1, 3) + 3 * h * h;
    A(1, 5) = A(1, 5) - 1.0;

    for (int i = 1; i < nodes.size() - 1; i++)
    {
        double h = std::get<0>(nodes[i]) - std::get<0>(nodes[i - 1]);

        A(2 + 4 * (i - 1) + 0, 4 * (i - 1) + 0) = 1.0;
        A(2 + 4 * (i - 1) + 0, 4 * (i - 1) + 1) = h;
        A(2 + 4 * (i - 1) + 0, 4 * (i - 1) + 2) = h * h;
        A(2 + 4 * (i - 1) + 0, 4 * (i - 1) + 3) = h * h * h;
        b(2 + 4 * (i - 1) + 0) = std::get<1>(nodes[i]);

        A(2 + 4 * (i - 1) + 1, 4 * i + 0) = 1.0;
        b(2 + 4 * (i - 1) + 1) = std::get<1>(nodes[i]);

        A(2 + 4 * (i - 1) + 2, 4 * (i - 1) + 1) = 1.0;
        A(2 + 4 * (i - 1) + 2, 4 * (i - 1) + 2) = 2 * h;
        A(2 + 4 * (i - 1) + 2, 4 * (i - 1) + 3) = 3 * h * h;
        A(2 + 4 * (i - 1) + 2, 4 * i + 1) = -1.0;
        b(2 + 4 * (i - 1) + 2) = 0.0;


        A(2 + 4 * (i - 1) + 3, 4 * (i - 1) + 2) = 2.0;
        A(2 + 4 * (i - 1) + 3, 4 * (i - 1) + 3) = 6 * h;
        A(2 + 4 * (i - 1) + 3, 4 * i + 2) = -2.0;
        b(2 + 4 * (i - 1) + 3) = 0.0;
    }

    for (size_t i = nodes.size() - 1; i < nodes.size(); i++)
    {
        double h = std::get<0>(nodes[i]) - std::get<0>(nodes[i - 1]);

        A(2 + 4 * (i - 1) + 0, 4 * (i - 1) + 0) = 1.0;
        A(2 + 4 * (i - 1) + 0, 4 * (i - 1) + 1) = h;
        A(2 + 4 * (i - 1) + 0, 4 * (i - 1) + 2) = h * h;
        A(2 + 4 * (i - 1) + 0, 4 * (i - 1) + 3) = h * h * h;
        b(2 + 4 * (i - 1) + 0) = std::get<1>(nodes[i]);

        A(2 + 4 * (i - 1) + 1, 4 * (i - 1) + 2) = 2.0;
        A(2 + 4 * (i - 1) + 1, 4 * (i - 1) + 3) = 6 * h;
        A(2 + 4 * (i - 1) + 1, 0) = 0;
    }

    for (size_t i = 2; i < N - 2; i += 4)
    {
        A.row(i + 1).swap(A.row(i + 3));

        double tmp = b(i + 3);
        b(i + 3) = b(i + 1);
        b(i + 1) = tmp;

        A.row(i + 2).swap(A.row(i + 3));

        double tmp1 = b(i + 3);
        b(i + 3) = b(i + 2);
        b(i + 2) = tmp1;
    }
}

std::vector<double> CalculateSplineCoefficients(const PointsVector& nodes)
{
    size_t N = 4 * (nodes.size() - 1);
    Eigen::MatrixXd A(N,N);
    Eigen::VectorXd b(N);
    A.setZero();
    b.setZero();

    InitializeMatricies(A, b, N, nodes);

    Eigen::PartialPivLU<Eigen::MatrixXd> lu(A);
    Eigen::VectorXd x = lu.solve(b);

    std::vector<double> Coefficients;
    for (size_t i = 0; i < N; i++)
    {
        Coefficients.push_back(x(i));
    }

    return Coefficients;
}

PointsVector Interpolate(const PointsVector& nodes, int step)
{
    PointsVector interpolatedPoints;

    for (int i = 0; i < nodes.size() - 1; i++)
    {
        double InterpolationStep = (std::get<0>(nodes[i + 1]) - std::get<0>(nodes[i])) / (step + 1);
        interpolatedPoints.push_back(nodes[i]);

        for (double j = std::get<0>(nodes[i]) + InterpolationStep; j < std::get<0>(nodes[i + 1])- InterpolationStep; j += InterpolationStep)
        {
#ifdef LaGarge
            interpolatedPoints.push_back({ j,LaGrangeInterpolation(j,nodes) });
#else
            interpolatedPoints.push_back({ j,SplineInterpolation(j,nodes,CalculateSplineCoefficients(nodes)) });
#endif
        }
    }

    return interpolatedPoints;
}

int main()
{
    constexpr uint32_t Interpolations = 13;
    constexpr uint32_t InterpolationsBetweenNodes = 30;
    const std::string FileName = "tczew_starogard";

    PointsVector FullData = LoadDataFromFile("2018_paths/" + FileName + ".txt");
	PointsVector PartialData;

    PartialData.push_back(FullData.front());

    size_t Step = FullData.size() / Interpolations;
	for (size_t i = Step; i < FullData.size(); i += Step)
	{
		PartialData.push_back(FullData[i]);
	}
    
    PartialData.push_back(FullData.back());

    PointsVector InterpolatedNodes = Interpolate(PartialData, InterpolationsBetweenNodes);

    ExportDataToFile(InterpolatedNodes, FileName + "Interpolated.txt");

#ifdef LaGarge
    std::string PythonCall = "python plot.py \"2018_paths/" + FileName + ".txt\" \"2018_paths/" + FileName + "Interpolated.txt\" \"" + FileName + "-LaGrange\"";
#else
    std::string PythonCall = "python plot.py \"2018_paths/" + FileName + ".txt\" \"2018_paths/" + FileName + "Interpolated.txt\" \"" + FileName + "-Spline\"";
#endif

    system(PythonCall.data());


	return 0;
}
