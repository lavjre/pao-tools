// Local Ghidra post-script used by Pao Tools headless decompilation.

import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileResults;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionIterator;
import ghidra.program.model.listing.FunctionManager;
import ghidra.program.model.listing.Parameter;
import ghidra.program.model.listing.Variable;

import java.io.BufferedWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public class PaoDecompile extends GhidraScript {
    private static String jsonEscape(String value) {
        if (value == null) {
            return "";
        }
        StringBuilder result = new StringBuilder();
        for (int index = 0; index < value.length(); index++) {
            char current = value.charAt(index);
            switch (current) {
                case '\\': result.append("\\\\"); break;
                case '"': result.append("\\\""); break;
                case '\n': result.append("\\n"); break;
                case '\r': result.append("\\r"); break;
                case '\t': result.append("\\t"); break;
                default:
                    if (current < 0x20) {
                        result.append(String.format("\\u%04x", (int) current));
                    }
                    else {
                        result.append(current);
                    }
            }
        }
        return result.toString();
    }

    private static String quoted(String value) {
        return "\"" + jsonEscape(value) + "\"";
    }

    private static String functionReference(Function function) {
        return "{\"name\":" + quoted(function.getName()) +
            ",\"entry\":" + quoted(function.getEntryPoint().toString()) + "}";
    }

    private static String functionSet(Set<Function> functions) {
        StringBuilder value = new StringBuilder("[");
        boolean first = true;
        for (Function function : functions) {
            if (!first) {
                value.append(',');
            }
            value.append(functionReference(function));
            first = false;
        }
        return value.append(']').toString();
    }

    private static String parameters(Parameter[] parameters) {
        StringBuilder value = new StringBuilder("[");
        for (int index = 0; index < parameters.length; index++) {
            if (index > 0) {
                value.append(',');
            }
            Parameter parameter = parameters[index];
            value.append("{\"ordinal\":").append(parameter.getOrdinal())
                .append(",\"name\":").append(quoted(parameter.getName()))
                .append(",\"type\":")
                .append(quoted(parameter.getDataType().getDisplayName()))
                .append(",\"storage\":")
                .append(quoted(parameter.getVariableStorage().toString()))
                .append('}');
        }
        return value.append(']').toString();
    }

    private static String localVariables(Variable[] variables) {
        StringBuilder value = new StringBuilder("[");
        for (int index = 0; index < variables.length; index++) {
            if (index > 0) {
                value.append(',');
            }
            Variable variable = variables[index];
            value.append("{\"name\":").append(quoted(variable.getName()))
                .append(",\"type\":")
                .append(quoted(variable.getDataType().getDisplayName()))
                .append(",\"storage\":")
                .append(quoted(variable.getVariableStorage().toString()))
                .append('}');
        }
        return value.append(']').toString();
    }

    private List<Function> selectFunctions(String selector) {
        FunctionManager manager = currentProgram.getFunctionManager();
        List<Function> selected = new ArrayList<>();
        if (selector == null || selector.isBlank() || selector.equalsIgnoreCase("all")) {
            FunctionIterator functions = manager.getFunctions(true);
            while (functions.hasNext()) {
                selected.add(functions.next());
            }
            return selected;
        }

        String addressText = selector.startsWith("0x") || selector.startsWith("0X")
            ? selector.substring(2)
            : selector;
        try {
            Address address = currentProgram.getAddressFactory().getAddress(addressText);
            if (address != null) {
                Function function = manager.getFunctionAt(address);
                if (function == null) {
                    function = manager.getFunctionContaining(address);
                }
                if (function != null) {
                    selected.add(function);
                    return selected;
                }
            }
        }
        catch (Exception ignored) {
            // Fall through to an exact function-name lookup.
        }

        FunctionIterator functions = manager.getFunctions(true);
        while (functions.hasNext()) {
            Function function = functions.next();
            if (function.getName().equals(selector)) {
                selected.add(function);
            }
        }
        return selected;
    }

    @Override
    public void run() throws Exception {
        String[] arguments = getScriptArgs();
        if (arguments.length < 4) {
            throw new IllegalArgumentException(
                "Expected: <output.c> <manifest.jsonl> <function|empty> <function-timeout>"
            );
        }

        Path outputPath = Path.of(arguments[0]);
        Path manifestPath = Path.of(arguments[1]);
        String selector = arguments[2];
        int functionTimeout = Integer.parseInt(arguments[3]);
        List<Function> functions = selectFunctions(selector);

        try (
            BufferedWriter output = Files.newBufferedWriter(outputPath, StandardCharsets.UTF_8);
            BufferedWriter manifest = Files.newBufferedWriter(manifestPath, StandardCharsets.UTF_8)
        ) {
            manifest.write(
                "{\"type\":\"program\",\"model_schema_version\":\"1\",\"name\":" + quoted(currentProgram.getName()) +
                ",\"format\":" + quoted(currentProgram.getExecutableFormat()) +
                ",\"language\":" + quoted(currentProgram.getLanguageID().toString()) +
                ",\"compiler\":" +
                quoted(currentProgram.getCompilerSpec().getCompilerSpecID().toString()) +
                ",\"image_base\":" + quoted(currentProgram.getImageBase().toString()) +
                ",\"function_count\":" + currentProgram.getFunctionManager().getFunctionCount() +
                ",\"symbol_count\":" + currentProgram.getSymbolTable().getNumSymbols() + "}\n"
            );

            if (functions.isEmpty()) {
                manifest.write(
                    "{\"type\":\"error\",\"status\":\"failed\",\"message\":" +
                    quoted("Function not found: " + selector) + "}\n"
                );
                return;
            }

            DecompInterface decompiler = new DecompInterface();
            decompiler.toggleCCode(true);
            decompiler.toggleSyntaxTree(true);
            if (!decompiler.openProgram(currentProgram)) {
                throw new IllegalStateException("Ghidra decompiler could not open the program");
            }
            try {
                for (Function function : functions) {
                    monitor.checkCancelled();
                    long started = System.currentTimeMillis();
                    DecompileResults result = decompiler.decompileFunction(
                        function, functionTimeout, monitor
                    );
                    long elapsed = System.currentTimeMillis() - started;
                    String name = function.getName();
                    String entry = function.getEntryPoint().toString();
                    if (result.decompileCompleted() && result.getDecompiledFunction() != null) {
                        output.write("/* Function: " + name + " @ " + entry + " */\n");
                        output.write(result.getDecompiledFunction().getC());
                        output.write("\n\n");
                        manifest.write(
                            "{\"type\":\"function\",\"status\":\"success\",\"name\":" +
                            quoted(name) + ",\"entry\":" + quoted(entry) +
                            ",\"namespace\":" + quoted(function.getParentNamespace().getName(true)) +
                            ",\"signature\":" + quoted(function.getPrototypeString(false, false)) +
                            ",\"calling_convention\":" + quoted(function.getCallingConventionName()) +
                            ",\"return_type\":" + quoted(function.getReturnType().getDisplayName()) +
                            ",\"parameters\":" + parameters(function.getParameters()) +
                            ",\"local_variables\":" + localVariables(function.getLocalVariables()) +
                            ",\"body_size\":" + function.getBody().getNumAddresses() +
                            ",\"external\":" + function.isExternal() +
                            ",\"thunk\":" + function.isThunk() +
                            ",\"var_args\":" + function.hasVarArgs() +
                            ",\"callers\":" + functionSet(function.getCallingFunctions(monitor)) +
                            ",\"callees\":" + functionSet(function.getCalledFunctions(monitor)) +
                            ",\"elapsed_ms\":" + elapsed + "}\n"
                        );
                    }
                    else {
                        manifest.write(
                            "{\"type\":\"function\",\"status\":\"failed\",\"name\":" +
                            quoted(name) + ",\"entry\":" + quoted(entry) +
                            ",\"elapsed_ms\":" + elapsed + ",\"error\":" +
                            quoted(result.getErrorMessage()) + "}\n"
                        );
                    }
                }
            }
            finally {
                decompiler.dispose();
            }
        }
    }
}
