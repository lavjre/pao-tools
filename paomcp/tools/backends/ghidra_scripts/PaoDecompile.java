// Local Ghidra post-script used by Pao Tools headless decompilation.

import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileResults;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionIterator;
import ghidra.program.model.listing.FunctionManager;

import java.io.BufferedWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

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
                "{\"type\":\"program\",\"name\":" + quoted(currentProgram.getName()) +
                ",\"format\":" + quoted(currentProgram.getExecutableFormat()) +
                ",\"language\":" + quoted(currentProgram.getLanguageID().toString()) +
                ",\"compiler\":" +
                quoted(currentProgram.getCompilerSpec().getCompilerSpecID().toString()) + "}\n"
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
